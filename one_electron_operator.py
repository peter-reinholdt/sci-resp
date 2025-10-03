import numba
import numpy as np

@numba.jit(nopython=True)
def arr_to_tuple(x):
    # only implemented for two words (up to 128 orbitals)
    if len(x) > 2:
        raise NotImplementedError
    if len(x) == 1:
        return (np.uint64(x[0]), np.uint64(0))
    else:
        return (np.uint64(x[0]), np.uint64(x[1]))


@numba.jit(nopython=True)
def bitcount(x, max_idx):
    b = 0
    for i in x[:max_idx]:
        while i > 0:
            i &= numba.uint64(i - 1)
            b += 1
    return b

@numba.jit(nopython=True)
def operator1e_state_multiplication(
    state: np.ndarray,
    h_int: float,
    ad_idx: int,
    a_idx: int,
    alpha_beta: str,
    idx2det: np.ndarray,
    det2idx: dict[tuple[tuple[int,...], tuple[int,...]], int],
) -> np.ndarray:
    num_dets = len(idx2det)
    new_state = np.zeros(num_dets)

    num = 64

    for i in range(num_dets):
        if np.abs(state[i])<1e-10:
            continue
        det_a, det_b = idx2det[i]
        if alpha_beta == "alpha":
            det = det_a.copy()
        else:
            det = det_b.copy()

        phase_changes = 0
        is_kill_state = False

        # Annihilate at a_idx
        nth_bit = (numba.uint64(det[a_idx//num]) >> numba.uint64(a_idx%num)) & 1
        if nth_bit == 1:
            tmp = det.copy()
            tmp[a_idx//num] = numba.uint64(det[a_idx//num]) & ((1 << numba.uint64(a_idx%num)) - 1)
            phase_changes += bitcount(tmp, ad_idx//num+1)
            det[a_idx//num] ^= numba.uint64(1) << numba.uint64(a_idx%num)
        else:
            is_kill_state = True

        # Create at ad_idx
        if not is_kill_state:
            nth_bit = (numba.uint64(det[ad_idx//num]) >> numba.uint64(ad_idx%num)) & 1
            if nth_bit == 0:
                tmp = det.copy()
                tmp[ad_idx//num] = numba.uint64(det[ad_idx//num]) & ((1 << numba.uint64(ad_idx%num)) - 1)
                phase_changes += bitcount(tmp, ad_idx//num+1)
                det[ad_idx//num] ^= numba.uint64(1) << numba.uint64(ad_idx%num)
            else:
                is_kill_state = True

        if not is_kill_state:
            if alpha_beta == "alpha":
                det_key = (arr_to_tuple(det), arr_to_tuple(det_b))
            else:
                det_key = (arr_to_tuple(det_a), arr_to_tuple(det))
            if det_key in det2idx:
                new_state[det2idx[det_key]] += h_int * (-1) ** phase_changes * state[i]
    return new_state


@numba.jit(nopython=True, parallel=True)
def singlet_operator_state(determinants, one_int, state):
    num_dets = len(determinants)
    num_orbs = len(one_int)

    det2idx = {}
    for i, (det_a, det_b) in enumerate(determinants):
        det2idx[(arr_to_tuple(det_a), arr_to_tuple(det_b))] = i

    nthreads = numba.get_num_threads()
    tmp = np.zeros((num_dets, nthreads))
    pqs = np.array([(p,q) for p in range(num_orbs) for q in range(num_orbs) if np.abs(one_int[p,q])>1e-14])
    for i in numba.prange(len(pqs)):
        p,q = pqs[i]
        thread_id = numba.get_thread_id()
        tmp[:,thread_id] += operator1e_state_multiplication(
            state, one_int[p, q], p, q, "alpha", determinants, det2idx
        )
        tmp[:,thread_id] += operator1e_state_multiplication(
            state, one_int[p, q], p, q, "beta", determinants, det2idx
        )
    new_state = np.sum(tmp, axis=1)
    return new_state


@numba.jit(nopython=True, parallel=True)
def triplet_operator_state(determinants, one_int, state):
    num_dets = len(determinants)
    num_orbs = len(one_int)

    det2idx = {}
    for i, (det_a, det_b) in enumerate(determinants):
        det2idx[(arr_to_tuple(det_a), arr_to_tuple(det_b))] = i

    nthreads = numba.get_num_threads()
    tmp = np.zeros((num_dets, nthreads))
    pqs = np.array([(p,q) for p in range(num_orbs) for q in range(num_orbs) if np.abs(one_int[p,q])>1e-14])
    for i in numba.prange(len(pqs)):
        p,q = pqs[i]
        thread_id = numba.get_thread_id()
        tmp[:,thread_id] += operator1e_state_multiplication(
            state, one_int[p, q], p, q, "alpha", determinants, det2idx
        )
        tmp[:,thread_id]  -= operator1e_state_multiplication(
            state, one_int[p, q], p, q, "beta", determinants, det2idx
        )
    new_state = np.sum(tmp, axis=1)
    return new_state
