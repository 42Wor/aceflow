from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union
from ..core.tensor import Tensor


class Parameter(Tensor):
    """A Tensor that represents a learnable parameter in a Module."""

    def __init__(self, data: Tensor):
        super().__init__(data, data.device)
        self.requires_grad = True


class Module:
    """Base class for all neural network modules in AceFlow."""

    def __init__(self) -> None:
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Parameter] = {}
        self.training = True

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if param is not None:
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional[Module]) -> None:
        if module is not None:
            self._modules[name] = module

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                super().__setattr__("_modules", {})
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> Iterator[Parameter]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def to(self, device: Union[str, Any]) -> Module:
        for name, param in self._parameters.items():
            self._parameters[name] = Parameter(param.to(device))
        for module in self._modules.values():
            module.to(device)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
