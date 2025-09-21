import builtins, re, inspect, numpy as np
_ORIG_RAND = np.random
def _blocked(*a, **k):
    raise RuntimeError("랜덤 데이터 생성 금지: metrics_real 사용하세요.")
# 배포 모드에서는 전면 차단
if not __debug__:
    np.random = None  # 완전 차단
else:
    # 개발 모드에서도 호출 시 에러
    class _R:
        def __getattr__(self, name):
            raise RuntimeError("np.random 금지: metrics_real을 호출하세요.")
    np.random = _R()
