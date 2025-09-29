import h5py
from gwpy.timeseries import TimeSeries
import os
import re
import ast


def _norm_key(k):
    return k.lower().replace('-', '_')


def find_value(cfg, possible_keys):
    """
    Recursively search cfg (a nested dict) for any key matching any of possible_keys.
    Keys are normalized (lower, '-' -> '_') to match variants like 'flow', 'flow', 'minimum-frequency', etc.
    """
    pk_norm = { _norm_key(k) for k in possible_keys }

    if not isinstance(cfg, dict):
        raise ValueError("Expected a dict to search in")

    for key, val in cfg.items():
        if _norm_key(key) in pk_norm:
            return val
        if isinstance(val, dict):
            try:
                return find_value(val, possible_keys)
            except ValueError:
                pass
    raise ValueError(f"No matching key found. Expected one of {possible_keys}")


def parse_freq_dict(freq_value):
    """
    Turn freq_value into a dict of {det: float} or {'waveform': float}.

    Accepts:
      - actual dict -> cast values to float
      - numeric scalar (int/float or numeric string) -> {'waveform': float}
      - python-dict-like string  -> ast.literal_eval -> dict
      - string of pairs (quoted or not) -> regex parse
      - otherwise -> {}
    """
    # already a dict
    if isinstance(freq_value, dict):
        return {str(k): float(v) for k, v in freq_value.items()}

    # numeric (int/float)
    if isinstance(freq_value, (int, float)):
        return {'waveform': float(freq_value)}

    # string handling
    if isinstance(freq_value, str):
        s = freq_value.strip()

        # try literal_eval for "{'H1': 20.0, ...}" or '{"H1": 20.0, ...}'
        if s.startswith('{') and s.endswith('}'):
            try:
                d = ast.literal_eval(s)
                if isinstance(d, dict):
                    return {str(k): float(v) for k, v in d.items()}
            except Exception:
                pass

        # try scalar numeric string
        try:
            val = float(s)
            return {'waveform': val}
        except Exception:
            pass

        # last resort: regex that allows quoted or unquoted keys
        pairs = re.findall(r"['\"]?([A-Za-z0-9_]+)['\"]?\s*:\s*([0-9.]+)", s)
        if pairs:
            return {k: float(v) for k, v in pairs}

    # nothing matched
    return {}


def result_config(result, app):
    """
    Robust extraction of waveform args and per-detector frequency ranges.
    """
    # support both dict-style `result['config']` and object-style `result.config`
    if hasattr(result, 'config'):
        cfg_root = result.config
    elif isinstance(result, dict) and 'config' in result:
        cfg_root = result['config']
    else:
        raise KeyError("result must have a .config attribute or a 'config' key")

    if app not in cfg_root:
        raise KeyError(f"App key '{app}' not found in result.config keys: {list(cfg_root.keys())}")

    config = cfg_root[app]

    # find variants
    fmin_val = find_value(config, ["fmin", "flow", "minimum_frequency", "minimum-frequency"])
    fmax_val = find_value(config, ["fmax", "fhigh", "maximum_frequency", "maximum-frequency"])
    fref_val = find_value(config, ["fref", "reference_frequency", "reference-frequency"])

    f_min_all = parse_freq_dict(fmin_val)
    f_max_all = parse_freq_dict(fmax_val)

    # f_ref can be scalar or dict-like; try to coerce
    f_ref_all = None
    try:
        if isinstance(fref_val, dict):
            f_ref_all = {str(k): float(v) for k, v in fref_val.items()}
        else:
            f_ref_all = float(fref_val)
    except Exception:
        # try parsing dict-like string
        try:
            parsed = ast.literal_eval(str(fref_val))
            if isinstance(parsed, dict):
                f_ref_all = {str(k): float(v) for k, v in parsed.items()}
            else:
                f_ref_all = float(parsed)
        except Exception:
            f_ref_all = str(fref_val)  # last resort: return raw

    # waveform minimum frequency (choose 'waveform' if given, else min across detectors, else fallback to f_ref)
    if "waveform" in f_min_all:
        f_min_wave = f_min_all["waveform"]
    elif f_min_all:
        f_min_wave = min(f_min_all.values())
    else:
        # final fallback: if f_ref_all is numeric, use it, otherwise None
        f_min_wave = float(f_ref_all) if isinstance(f_ref_all, (int, float)) else None

    approximant = app.split(':')[1]
    delta_t = float(config.get('delta_t', 1.0 / 4096.0)) if isinstance(config, dict) else 1.0 / 4096.0

    args = {
        'approximant': approximant,
        'delta_t': delta_t,
        'f_low': f_min_wave,
        'f_ref': f_ref_all,
    }

    ST_flags = {'PhenomXPFinalSpinMod' : 2,
    'PhenomXPrecVersion' : 320,
    'PhenomXHMReleaseVersion' : 122022}

    if approximant == 'IMRPhenomXPHM-SpinTaylor':
        args['approximant'] = "IMRPhenomXPHM"
        args['flags'] = ST_flags 

    # build per-detector det_args
    det_args = {}
    for det in sorted(set(f_min_all.keys()) | set(f_max_all.keys())):
        det_args[det] = {
            'f_min': f_min_all.get(det),
            'f_max': f_max_all.get(det)
        }

    return args, det_args