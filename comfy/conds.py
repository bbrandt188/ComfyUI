import mlx.core as mx
import math
import comfy.utils


def lcm(a, b):  # TODO: replace by math.lcm when using Python 3.9+
    return abs(a * b) // math.gcd(a, b)

class CONDRegular:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(comfy.utils.repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return mx.concatenate(conds)

class CONDNoiseShape(CONDRegular):
    def process_cond(self, batch_size, device, area, **kwargs):
        data = self.cond
        if area is not None:
            dims = len(area) // 2
            for i in range(dims):
                data = mx.narrow(data, i + 2, area[i + dims], area[i])

        return self._copy_with(comfy.utils.repeat_to_batch_size(data, batch_size).to(device))

class CONDCrossAttn(CONDRegular):
    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:  # these cases should not happen
                return False

            mult_min = lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if diff > 4:  # Arbitrary limit to avoid performance hits due to excessive padding
                return False
        return True

    def concat(self, others):
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = mx.repeat(c, (1, crossattn_max_len // c.shape[1], 1))  # Padding with repeat
            out.append(c)
        return mx.concatenate(out)

class CONDConstant(CONDRegular):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(self.cond)

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond
