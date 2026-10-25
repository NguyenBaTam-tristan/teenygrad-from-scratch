"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    # TODO: Multiply together the elements of a shape tuple to get the total number of elements.
    prod = 1
    if not shape: return 1
    for num in shape:
        prod *= num
    return prod

# Step 2 - argsort
def argsort(values):
    # TODO: Return the indices that would sort values in ascending order.
    index_values = list(enumerate(values))
    sorted_values = sorted(index_values , key=lambda x: x[1])
    result = []
    for i in range(len(sorted_values)):
        result.append(sorted_values[i][0])
    return result

# Step 3 - make_op_enums
from enum import Enum, auto
def make_op_enums():
    # TODO: create four enum classes naming every supported operation kind
    # phép toán một ngôi: nhận 1 tensor, biến dổi từng 
    class UnaryOps(Enum):
        NEG = auto() # đổi dấu x -> -x
        RELU = auto() # hàm kích hoạt Relu: max(0, x)
        LOG = auto() # logarit tự nhiên (ln(x))
        EXP = auto() # hàm mũ cơ số e (e^x)
        SQRT = auto() # căn bậc 2
        SIGMOID = auto() # 1 / (1 + (e^(-x)))
    # phép toán 2 ngôi: nhận 2 tensor cùng or khác kích thước
    class BinaryOps(Enum):
        ADD = auto() # cộng
        SUB = auto() # trừ
        MUL = auto() # nhân
        DIV = auto() # chia
        CMPLT = auto() # so sánh return True False
        MAX = auto() # max(a, b) từng cặp
    # thu gọn chiều theo trục chỉ định
    class ReduceOps(Enum):
        SUM = auto() # tổng theo trục
        MAX = auto() # max theo trục
    # biến dổi không gian
    class MovementOps(Enum):
        RESHAPE = auto() # đỏi kích thước từ (2,3) -> (6,)
        EXPAND = auto() # kéo một kích thước độ dài 1 thành kích thước lớn hơn cho broadcast
        PERMUTE = auto() # tráo đổi thứ tự các trục
    
    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# Step 4 - LazyBuffer
class LazyBuffer:
    def __init__(self, np_array):
        # TODO: wrap np_array as an ndarray and expose shape and dtype
        if isinstance(np_array, np.ndarray):
            self._np = np_array
        else:
            self._np = np.array(np_array)
        
        self.shape = self._np.shape
        self.dtype = self._np.dtype

    def tolist(self):
        return self._np.tolist()

    def __repr__(self):
        return f'LazzyBuffer shape = {self.shape} dtype = {self.dtype}'

# Step 5 - lazybuffer_const
def const(value, shape):
    # TODO: Create a new LazyBuffer of the given shape filled with a constant value.
    arr = np.full(shape, value, dtype=np.float32)
    return LazyBuffer(arr)
LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    # TODO: return a LazyBuffer of uniform random floats in [0, 1) with given shape
    rng = np.random.default_rng(seed)
    arr = rng.random(shape, dtype=np.float32)
    return LazyBuffer(arr)

# Step 7 - lazybuffer_unary_e
UnaryOps, _, _, _ = make_op_enums()
def e(self, op):
    # TODO: apply a unary elementwise op (NEG, RELU, LOG, EXP, SQRT, SIGMOID)
    # lấy ra class unaryOps từ hàm make_op_enums đã định nghĩa 
    data = self._np
    # ánh xạ từng phép toán tương ứng từ hàm numpy
    if op == UnaryOps.NEG:
        res = -data
    elif op == UnaryOps.RELU:
        res = np.maximum(0, data)
    elif op == UnaryOps.LOG:
        res = np.log(data)
    elif op == UnaryOps.EXP:
        res = np.exp(data)
    elif op == UnaryOps.SQRT:
        res = np.sqrt(data)
    elif op == UnaryOps.SIGMOID:
        res = 1 / (1 + np.exp(-data))
    else:
        raise ValueError(f"Unsupported unary op: {op}")
    
    return LazyBuffer(res)
LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
_, BinaryOps, _, _ = make_op_enums()
def lazybuffer_binary_e(self, op, other):
    # TODO: apply a binary elementwise op between two LazyBuffers, return a new LazyBuffer
    x = self._np
    y = other._np

    if op == BinaryOps.ADD:
        res = x+y
    elif op == BinaryOps.SUB:
        res = x-y
    elif op == BinaryOps.MUL:
        res = x*y 
    elif op == BinaryOps.DIV:
        res = x/y 
    elif op == BinaryOps.CMPLT:
        res = (x < y).astype(np.float32)
    elif op == BinaryOps.MAX:
        res = np.maximum(x, y)
    else:
        raise ValueError(f"Unsupported binary op: {op}")
    
    return LazyBuffer(res)

# Step 9 - lazybuffer_r
_, _, ReduceOps, _ = make_op_enums()
def r(self, op, axis):
    # TODO: reduce the underlying array along axis (SUM or MAX), keeping reduced dims as size 1
    data = self._np

    if op == ReduceOps.SUM:
        res = np.sum(data, axis=axis, keepdims = True)
    elif op == ReduceOps.MAX:
        res = np.max(data, axis=axis, keepdims = True)
    else:
        raise ValueError(f"Unsupported reduce op: {op}")
    return LazyBuffer(res)
LazyBuffer.r=r

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    # TODO: return a new LazyBuffer with the array reshaped to new_shape
    reshaped_data = self._np.reshape(new_shape)
    return LazyBuffer(reshaped_data)
    lazyBuffer.reshape = reshape

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    # TODO: broadcast this buffer's size-1 dims out to new_shape
    expanded_data = np.broadcast_to(self._np, new_shape)
    return LazyBuffer(expanded_data)
LazyBuffer.expand = expand

# Step 12 - lazybuffer_permute
def permute(self, order):
    # TODO: return a new LazyBuffer with axes reordered according to order
    permuted_data = np.transpose(self._np, axes = order)
    return LazyBuffer(permuted_data)
LazyBuffer.permute = permute

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        # 1. Ghi nhận trực tiếp cờ requires_grad của từng tensor (giữ nguyên True, False, hoặc None)
        self.needs_input_grad = [t.requires_grad for t in tensors]
        
        # 2. Xử lý 3 trạng thái: True > None > False
        if any(x is True for x in self.needs_input_grad):
            self.requires_grad = True
        elif any(x is None for x in self.needs_input_grad):
            self.requires_grad = None
        else:
            self.requires_grad = False

        # 3. CHỈ tạo thuộc tính parents khi gradient thực sự truyền qua (True)
        # Không tạo nhánh else để hasattr(f, 'parents') trả về False khi không cần grad
        if self.requires_grad:
            self.parents = tensors

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    # TODO: attach forward and backward stubs to Function that raise NotImplementedError
    def forward(self, *arg, **kwargs):
        raise NotImplementedError
    def backward(self, *arg, **kwargs):
        raise NotImplementedError
    Function.forward = forward
    Function.backward = backward

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    # TODO: build the Function, run forward on the input buffers, wrap in a
    # Tensor, and link out._ctx when a gradient is needed.
    # Chuẩn bị ghi chép lý lịch các tensor.
    ctx = cls(*tensors)
    # Bóc lõi dữ liệu thô lazydata để thực hiện phép toán thực tế.
    ret = ctx.forward(*[t.lazydata for t in tensors], **kwargs)
    # Đóng gói kết quả thô thành Tensor mới và kế thừa cờ đạo hàm.
    out = Tensor(ret, requires_grad = ctx.requires_grad)
    # Móc sợi dây liên kết ngược để lưu lại nguồn gốc phục vụ backward.
    if ctx.requires_grad:
        out._ctx = ctx
    return out

# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
import numpy as np
class Neg(Function):
    def forward(self, x):
        # TODO: return a LazyBuffer holding the elementwise negation of x
        return x.e(UnaryOps.NEG)

    def backward(self, grad_output):
        # TODO: return the negated incoming gradient
        return grad_output.e(UnaryOps.NEG)

# Step 17 - Relu
class Relu(Function):
    def forward(self, x):
        # TODO: apply the rectified linear unit to lazy buffer x and cache the result
        self.ret = x.e(UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output):
        # TODO: route the upstream gradient only through positions that were positive
        zero = const(0.0, self.ret.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.ret)
        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, mask)

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

