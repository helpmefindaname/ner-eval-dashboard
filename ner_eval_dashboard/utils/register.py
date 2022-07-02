from typing import Type, Dict, List, TypeVar


class RegisterMixin:
    _registered_classes: Dict[str, Type] = {}
    registered_names: List[str] = []

    @classmethod
    def register(cls, name: str):
        def inner_register(register_cls: Type) -> Type:
            cls._registered_classes[name] = register_cls
            cls.registered_names.append(name)
            return register_cls

        return inner_register

    @classmethod
    def load(cls, name: str) -> Type:
        return cls._registered_classes[name]


T = TypeVar("T", bound=RegisterMixin)


def setup_register(cls: T) -> T:
    cls._registered_classes = {}
    cls.registered_names = []
    return cls
