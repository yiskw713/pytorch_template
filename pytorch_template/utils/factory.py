import inspect
from abc import ABCMeta
from typing import Any, Callable, Generic, Type, TypeVar

T = TypeVar("T")


class Factory(Generic[T], metaclass=ABCMeta):
    """Factory Base Class."""

    class_dict: dict[str, Type[T]] = {}

    def __init_subclass__(cls) -> None:
        """Create new dictionary when subclass is defined."""
        cls.class_dict = {}
        return super().__init_subclass__()

    @classmethod
    def get_registered_classes(cls) -> dict[str, Type[T]]:
        """Return registered classes in form of dictionary.

        Returns:
            dict[str, Type[T]]: The pair of registered name and class.
        """
        return cls.class_dict

    @classmethod
    def register_to_dict(cls, name: str, class_type: Type[T]) -> None:
        """Registering new class to the `class_dict`.

        Args:
            name (str): Registered name.
            class_type (Type[T]): Class type.

        Raises:
            ValueError: Error will be raised if `name` has already been in `class_dict`.
        """
        if name in cls.class_dict:
            raise ValueError(f"Name: {name} has already been registered.")

        cls.class_dict[name] = class_type

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator for registering new class to the factory.

        Args:
            name (str): The class name.

        Returns:
            Callable[[Type[T]], Type[T]]: Wrapped method.
        """

        def wrapper(class_type: Type[T]) -> Type[T]:
            cls.register_to_dict(name, class_type)
            return class_type

        return wrapper

    @classmethod
    def register_from_module(cls, module: Any) -> None:
        """Register all of the classes defined in `module`.

        Args:
            module (Any): Module (e.g. torch.optim)
        """
        # filtering only class types from `module`.
        class_names = [x for x in dir(module) if isinstance(getattr(module, x), type)]

        for name in class_names:
            cls.register_to_dict(name, getattr(module, name))

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> T:
        """Create an instance of the specified class.

        Args:
            name (str): Class name.

        Raises:
            ValueError: Error will be raised if the name is invalid.

        Returns:
            T: Instantiated object.
        """
        if name not in cls.class_dict:
            raise ValueError(
                f"Invalid name `{name}`.\n"
                f"Registered classes are: {cls.class_dict.values()}"
            )

        class_type = cls.class_dict[name]
        parameters = cls.filter_required_arguments(class_type, kwargs)
        return class_type(*args, **parameters)

    @staticmethod
    def filter_required_arguments(
        class_type: Type[T], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Filtering required arguments when creating an instance of `class_type`

        Args:
            class_type (Type[Type[T]]): Class type.
            kwargs (dict[str, Any]): Keyword arguments.

        Returns:
            dict[str, Any]: Filtered keyword arguments.
        """
        signature = inspect.signature(class_type)  # type: ignore
        return {k: kwargs.get(k, v.default) for k, v in signature.parameters.items()}
