import torch
import torch.nn as nn
from collections import deque


class TopologyManagerV2:
    """
    Policy-driven topology + geometry controller for Mnemonic Cortex.
    """

    def __init__(self, mutation_rate=0.01, default_policy="default", ema_beta=0.9):
        self.mutation_rate = float(mutation_rate)
        self.policies = {}
        self.active_policy = None
        self.ema_beta = float(ema_beta)
        self.fitness_ema = None
        self.fitness_hist = deque(maxlen=50)
        self._managed_mergers = []

        self.register_policy(
            default_policy,
            micro_b=0.02,
            micro_b_max=0.05,
            omega_max=0.20,
            temp_scale=1.0,
            use_heat_kernel=True,
            allowed_channels=None,
            gate_bias=None,
            curvature_mode="mix",
            curvature_rate=mutation_rate,
            curvature_mix=(0.4, 0.3, 0.2, 0.1),
            qhm_enable=True,
            qhm_alpha_override=None,
            qhm_temp=1.0,
            qhm_phase_noise=0.0,
            qhm_lightbulb=0.92,
            qhm_explosive_temp=0.6,
            qhm_explosive_alpha=0.6,
            qhm_lb_z_on=None,
            qhm_lb_z_off=None,
            qhm_lb_cooldown_steps=None,
            qhm_lb_max_stage2_steps=None,
            qhm_lb_budget_per_100=None,
            qhm_lb_prefocus_temp_mult=None,
            qhm_lb_prefocus_alpha_boost=None,
            qhm_lb_alpha_max=None,
            qhm_lb_temp_min=None,
            adaptive=dict(
                b_lo=0.01,
                b_hi=0.03,
                omega_lo=0.10,
                omega_hi=0.25,
                fit_up=0.80,
                fit_down=0.55,
                step=0.002,
            ),
        ).activate_policy(default_policy)

    # ------------------------- Policy API -------------------------

    def register_policy(
        self,
        name,
        micro_b=0.02,
        micro_b_max=0.05,
        omega_max=0.20,
        temp_scale=1.0,
        use_heat_kernel=True,
        allowed_channels=None,
        gate_bias=None,
        curvature_mode="mix",
        curvature_rate=0.01,
        curvature_mix=(0.4, 0.3, 0.2, 0.1),
        qhm_enable=True,
        qhm_alpha_override=None,
        qhm_temp=1.0,
        qhm_phase_noise=0.0,
        qhm_lightbulb=0.92,
        qhm_explosive_temp=0.6,
        qhm_explosive_alpha=0.6,
        qhm_lb_z_on=None,
        qhm_lb_z_off=None,
        qhm_lb_cooldown_steps=None,
        qhm_lb_max_stage2_steps=None,
        qhm_lb_budget_per_100=None,
        qhm_lb_prefocus_temp_mult=None,
        qhm_lb_prefocus_alpha_boost=None,
        qhm_lb_alpha_max=None,
        qhm_lb_temp_min=None,
        adaptive=None,
    ):
        policy = dict(
            micro_b=float(micro_b),
            micro_b_max=float(micro_b_max),
            omega_max=float(omega_max),
            temp_scale=float(temp_scale),
            use_heat_kernel=bool(use_heat_kernel),
            allowed_channels=None
            if allowed_channels is None
            else torch.as_tensor(allowed_channels, dtype=torch.float32),
            gate_bias=None if gate_bias is None else torch.as_tensor(gate_bias, dtype=torch.float32),
            curvature_mode=str(curvature_mode),
            curvature_rate=float(curvature_rate),
            curvature_mix=tuple(float(x) for x in curvature_mix),
            qhm_enable=bool(qhm_enable),
            qhm_alpha_override=qhm_alpha_override,
            qhm_temp=float(qhm_temp),
            qhm_phase_noise=float(qhm_phase_noise),
            qhm_lightbulb=float(qhm_lightbulb),
            qhm_explosive_temp=float(qhm_explosive_temp),
            qhm_explosive_alpha=float(qhm_explosive_alpha),
            qhm_lb_z_on=None if qhm_lb_z_on is None else float(qhm_lb_z_on),
            qhm_lb_z_off=None if qhm_lb_z_off is None else float(qhm_lb_z_off),
            qhm_lb_cooldown_steps=None
            if qhm_lb_cooldown_steps is None
            else int(qhm_lb_cooldown_steps),
            qhm_lb_max_stage2_steps=None
            if qhm_lb_max_stage2_steps is None
            else int(qhm_lb_max_stage2_steps),
            qhm_lb_budget_per_100=None
            if qhm_lb_budget_per_100 is None
            else int(qhm_lb_budget_per_100),
            qhm_lb_prefocus_temp_mult=None
            if qhm_lb_prefocus_temp_mult is None
            else float(qhm_lb_prefocus_temp_mult),
            qhm_lb_prefocus_alpha_boost=None
            if qhm_lb_prefocus_alpha_boost is None
            else float(qhm_lb_prefocus_alpha_boost),
            qhm_lb_alpha_max=None if qhm_lb_alpha_max is None else float(qhm_lb_alpha_max),
            qhm_lb_temp_min=None if qhm_lb_temp_min is None else float(qhm_lb_temp_min),
            adaptive=dict(
                b_lo=0.01,
                b_hi=0.03,
                omega_lo=0.10,
                omega_hi=0.25,
                fit_up=0.80,
                fit_down=0.55,
                step=0.002,
            )
            if adaptive is None
            else adaptive,
        )
        self.policies[name] = policy
        return self

    def activate_policy(self, name, model=None):
        assert name in self.policies, f"Unknown policy '{name}'"
        self.active_policy = name
        if model is not None:
            self.apply_to_model(model)
        return self

    def current_policy(self):
        return self.policies[self.active_policy]

    # ------------------- Push knobs into mergers -------------------

    @torch.no_grad()
    def _apply_policy_to_merger(self, merger: nn.Module, policy: dict):
        b = min(policy["micro_b"], policy["micro_b_max"])
        if hasattr(merger, "set_micro_b"):
            merger.set_micro_b(b)
        if hasattr(merger, "set_omega_max"):
            merger.set_omega_max(policy["omega_max"])
        if hasattr(merger, "use_heat"):
            merger.use_heat = bool(policy["use_heat_kernel"])
        if hasattr(merger, "set_temp_scale"):
            merger.set_temp_scale(policy["temp_scale"])

        if hasattr(merger, "set_gate_bias"):
            if policy["allowed_channels"] is not None:
                mask = policy["allowed_channels"].to(merger.gate_bias.device)
                hard_bias = torch.where(
                    mask > 0.5, torch.zeros_like(mask), torch.full_like(mask, -12.0)
                )
            else:
                hard_bias = torch.zeros(6, device=merger.gate_bias.device)

            if policy["gate_bias"] is not None:
                gb = policy["gate_bias"].to(hard_bias.device)
                merger.set_gate_bias(hard_bias + gb)
            else:
                merger.set_gate_bias(hard_bias)

    @torch.no_grad()
    def _apply_qhm_to_mem(self, mem, policy: dict):
        if mem is None or not hasattr(mem, "holo"):
            return
        mem.qhm_enabled = bool(policy.get("qhm_enable", True))
        mem.holo.set_temperature(policy.get("qhm_temp", 1.0))
        mem.holo.set_phase_noise(policy.get("qhm_phase_noise", 0.0))
        mem.holo.set_lightbulb(
            policy.get("qhm_lightbulb", 0.92),
            policy.get("qhm_explosive_temp", 0.6),
            policy.get("qhm_explosive_alpha", 0.6),
        )
        if hasattr(mem, "lb_ctrl"):
            z_on = policy.get("qhm_lb_z_on", None)
            z_off = policy.get("qhm_lb_z_off", None)
            if z_on is not None or z_off is not None:
                mem.lb_ctrl.set_thresholds(
                    mem.lb_ctrl.z_on if z_on is None else float(z_on),
                    z_off,
                )
            cd = policy.get("qhm_lb_cooldown_steps", None)
            s2 = policy.get("qhm_lb_max_stage2_steps", None)
            bud = policy.get("qhm_lb_budget_per_100", None)
            if cd is not None or s2 is not None or bud is not None:
                mem.lb_ctrl.set_budget(
                    mem.lb_ctrl.cooldown_steps if cd is None else int(cd),
                    mem.lb_ctrl.max_stage2_steps if s2 is None else int(s2),
                    mem.lb_ctrl.budget_per_100 if bud is None else int(bud),
                )
            pt = policy.get("qhm_lb_prefocus_temp_mult", None)
            pa = policy.get("qhm_lb_prefocus_alpha_boost", None)
            if pt is not None or pa is not None:
                mem.lb_ctrl.set_prefocus(
                    mem.lb_ctrl.prefocus_temp_mult if pt is None else float(pt),
                    mem.lb_ctrl.prefocus_alpha_boost if pa is None else float(pa),
                )
            mem.lb_ctrl.set_explosive(
                policy.get("qhm_explosive_temp", 0.6),
                policy.get("qhm_explosive_alpha", 0.6),
            )
            amax = policy.get("qhm_lb_alpha_max", None)
            tmin = policy.get("qhm_lb_temp_min", None)
            if amax is not None or tmin is not None:
                mem.lb_ctrl.set_limits(
                    mem.lb_ctrl.alpha_max if amax is None else float(amax),
                    mem.lb_ctrl.temp_min if tmin is None else float(tmin),
                )
        mem.qhm_alpha_override = policy.get("qhm_alpha_override", None)

    @torch.no_grad()
    def apply_to_model(self, model):
        self._managed_mergers.clear()

        def maybe_add(obj):
            if obj is None:
                return
            if hasattr(obj, "geometry_merger"):
                gm = getattr(obj, "geometry_merger")
                if hasattr(gm, "set_micro_b") and hasattr(gm, "set_omega_max"):
                    self._managed_mergers.append(gm)

        maybe_add(getattr(model, "working_memory", None))

        ltm = getattr(model, "long_term_memory", None)
        if ltm is not None:
            maybe_add(getattr(ltm, "hyper_geometric", None))
            maybe_add(getattr(ltm, "hg", None))
            maybe_add(getattr(ltm, "cgmn", None))
            maybe_add(getattr(ltm, "curved", None))

        pol = self.current_policy()
        for merger in self._managed_mergers:
            self._apply_policy_to_merger(merger, pol)
        self._apply_qhm_to_mem(getattr(model, "working_memory", None), pol)
        if ltm is not None:
            self._apply_qhm_to_mem(getattr(ltm, "hyper_geometric", None), pol)
            self._apply_qhm_to_mem(getattr(ltm, "hg", None), pol)
            self._apply_qhm_to_mem(getattr(ltm, "cgmn", None), pol)
            self._apply_qhm_to_mem(getattr(ltm, "curved", None), pol)
        return self

    # ----------------- Adaptive tweak by fitness ------------------

    def update_fitness(self, loss_value: float):
        fitness = 1.0 / (1.0 + float(loss_value))
        self.fitness_hist.append(fitness)
        if self.fitness_ema is None:
            self.fitness_ema = fitness
        else:
            self.fitness_ema = self.ema_beta * self.fitness_ema + (1.0 - self.ema_beta) * fitness
        return self.fitness_ema

    @torch.no_grad()
    def adapt_policy_bounds(self):
        pol = self.current_policy()
        a = pol["adaptive"]
        fit = self.fitness_ema if self.fitness_ema is not None else 0.0
        pol = dict(pol)
        step = float(a["step"])

        b = float(pol["micro_b"])
        if fit >= a["fit_up"]:
            b = max(a["b_lo"], b - step)
        elif fit <= a["fit_down"]:
            b = min(a["b_hi"], b + step)
        pol["micro_b"] = float(min(b, pol["micro_b_max"]))

        om = float(pol["omega_max"])
        if fit >= a["fit_up"]:
            om = max(a["omega_lo"], om - step * 2.0)
        elif fit <= a["fit_down"]:
            om = min(a["omega_hi"], om + step * 2.0)
        pol["omega_max"] = float(om)

        self.policies[self.active_policy] = pol
        for merger in self._managed_mergers:
            self._apply_policy_to_merger(merger, pol)

    # ---------------- Curvature mutation (safe) -------------------

    @torch.no_grad()
    def mutate_curvature(self, curvature: torch.Tensor):
        pol = self.current_policy()
        mode = pol["curvature_mode"]
        rate = float(pol["curvature_rate"])

        if curvature.dim() == 2:
            slot_scalar = curvature.mean(dim=1)
            expand_back = lambda s: s.unsqueeze(-1).expand_as(curvature)
        elif curvature.dim() == 1:
            slot_scalar = curvature
            expand_back = lambda s: s
        else:
            raise ValueError("curvature must be [M] or [M,D]")

        def do_mut(s, kind):
            if kind == "hyperbolic":
                return s + rate * torch.randn_like(s)
            if kind == "spherical":
                return s - rate * torch.abs(torch.randn_like(s))
            if kind == "euclidean":
                return rate * torch.randn_like(s)
            if kind == "fractal":
                s_f = torch.fft.rfft(s.float())
                noise = torch.randn_like(s_f.real) * rate
                s_new = torch.fft.irfft(s_f + noise, n=s.numel())
                return s_new.to(s.dtype)
            return s

        if mode == "mix":
            w_h, w_s, w_e, w_f = pol["curvature_mix"]
            s_h = do_mut(slot_scalar, "hyperbolic")
            s_s = do_mut(slot_scalar, "spherical")
            s_e = do_mut(slot_scalar, "euclidean")
            s_f = do_mut(slot_scalar, "fractal")
            slot_new = (w_h * s_h + w_s * s_s + w_e * s_e + w_f * s_f) / (
                w_h + w_s + w_e + w_f + 1e-9
            )
        else:
            slot_new = do_mut(slot_scalar, mode)

        return expand_back(slot_new)

    # ------------------- One-call training hook -------------------

    @torch.no_grad()
    def step(self, model, loss_value: float):
        self.update_fitness(loss_value)
        self.adapt_policy_bounds()

        def maybe_mutate(mem):
            if mem is None:
                return
            for attr in ("curvature", "memory_curvature"):
                if hasattr(mem, attr):
                    par = getattr(mem, attr)
                    if isinstance(par, torch.nn.Parameter):
                        par.data.copy_(self.mutate_curvature(par.data))

        ltm = getattr(model, "long_term_memory", None)
        if ltm is not None:
            maybe_mutate(getattr(ltm, "hyper_geometric", None))
            maybe_mutate(getattr(ltm, "hg", None))
            maybe_mutate(getattr(ltm, "cgmn", None))

        self.apply_to_model(model)

    def list_policies(self):
        return list(self.policies.keys())

    def clone_policy(self, src_name: str, dst_name: str, **overrides):
        assert src_name in self.policies, f"Unknown policy '{src_name}'"
        p = dict(self.policies[src_name])
        p.update(overrides)
        self.policies[dst_name] = p
        return self

    def register_babi_qhm_v2_policy(self, name: str = "babi_qhm_v2"):
        """
        Register a ready-to-use policy tuned for bAbI-like retrieval workloads.

        Usage:
            model.topology.register_babi_qhm_v2_policy()
            model.apply_topology_policy("babi_qhm_v2")
        """
        return self.register_policy(
            name,
            micro_b=0.018,
            micro_b_max=0.05,
            omega_max=0.18,
            temp_scale=1.05,
            use_heat_kernel=True,
            allowed_channels=[1, 1, 1, 1, 0, 0],
            gate_bias=[0.4, 0.2, 0.2, 0.1, -2.0, -2.0],
            curvature_mode="spherical",
            curvature_rate=0.006,
            qhm_enable=True,
            qhm_temp=0.90,
            qhm_phase_noise=0.03,
            qhm_lightbulb=0.93,
            qhm_explosive_temp=0.55,
            qhm_explosive_alpha=0.65,
            qhm_alpha_override=None,
            qhm_lb_z_on=2.0,
            qhm_lb_z_off=1.2,
            qhm_lb_cooldown_steps=6,
            qhm_lb_max_stage2_steps=2,
            qhm_lb_budget_per_100=6,
            qhm_lb_prefocus_temp_mult=0.87,
            qhm_lb_prefocus_alpha_boost=0.12,
            qhm_lb_alpha_max=0.85,
            qhm_lb_temp_min=0.45,
        )
