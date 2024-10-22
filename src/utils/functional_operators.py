from typing import Union
import numpy as np


MIN_VALUE = np.finfo(np.float64).min
MAX_VALUE = np.finfo(np.float64).max

# Максимальное значение x, для которого exp(x) не вызовет переполнения
MAX_EXP_INPUT = np.log(MAX_VALUE)


def save_div(x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    result: Union[float, np.ndarray]
    if isinstance(y, np.ndarray):
        result = np.divide(x, y, out=np.ones_like(y, dtype=np.float64), where=y != 0)
    else:
        if y == 0:
            result = 0.0
        else:
            result = x / y
    result = np.clip(result, MIN_VALUE, MAX_VALUE)
    return result


def logabs(y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    y_ = np.abs(y)
    if isinstance(y_, np.ndarray):
        result = np.log(y_, out=np.ones_like(y_, dtype=np.float64), where=y_ != 0)
    else:
        if y_ == 0:
            result = 1
        else:
            result = np.log(y_)
    return result


def save_exp(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # Ограничиваем входные значения x
    x_clipped = np.clip(x, MIN_VALUE, MAX_EXP_INPUT)
    result = np.exp(x_clipped)
    return result


def sqrtabs(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    result = np.sqrt(np.abs(x))
    return result


def cos(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.cos(x)


def sin(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.sin(x)


class Operator:
    def __init__(self, formula: str, name: str, sign: str, operation: callable, arity: int) -> None:
        self.formula = formula
        self.__name__ = name or None
        self.sign = sign
        self.operation = operation
        self.arity = arity

    def write(self, *args: any) -> str:
        return self.formula.format(*args)

    def __call__(self, *args: any) -> any:
        return self.operation(*args)


def create_operator(formula: str, name: str, sign: str, operation: callable, arity: int) -> Operator:
    return Operator(formula, name, sign, operation, arity)
