package io.crystalforge.minecraft;

public final class CrystalForgeRequests {
    private CrystalForgeRequests() {
    }

    public static CrystalForgeWorkflowRequest tfimQuantumEscalation() {
        return new CrystalForgeWorkflowRequest("""
            {
              "model_family": "tfim",
              "Lx": 2,
              "Ly": 2,
              "parameters": { "J": 1.0, "h": 0.8, "g": 0.0 },
              "qprobe_targets": ["Mz", "ZZ_nn", "Mstag2"],
              "qprobe_tolerance": 0.03,
              "qprobe_shots_per_group": 4000,
              "qprobe_readout_flip_prob": 0.02,
              "qprobe_seed": 7
            }
            """);
    }

    public static CrystalForgeWorkflowRequest tfimSafe() {
        return new CrystalForgeWorkflowRequest("""
            {
              "model_family": "tfim",
              "Lx": 2,
              "Ly": 2,
              "parameters": { "J": 1.0, "h": 0.1, "g": 1.4 },
              "qprobe_targets": ["Mz", "ZZ_nn", "Mstag2"],
              "qprobe_tolerance": 0.03,
              "qprobe_shots_per_group": 4000,
              "qprobe_readout_flip_prob": 0.02,
              "qprobe_seed": 7
            }
            """);
    }

    public static CrystalForgeWorkflowRequest hubbardFallback() {
        return new CrystalForgeWorkflowRequest("""
            {
              "model_family": "hubbard",
              "Lx": 2,
              "Ly": 2,
              "parameters": { "t": 1.0, "U": 4.0, "mu": 2.0 },
              "qprobe_targets": ["D", "Ms2", "Cs_max"],
              "qprobe_tolerance": 0.03,
              "qprobe_shots_per_group": 2000,
              "qprobe_readout_flip_prob": 0.01,
              "qprobe_seed": 7
            }
            """);
    }

    public static CrystalForgeWorkflowRequest fromPreset(String name) {
        return switch (name) {
            case "tfim_safe" -> tfimSafe();
            case "tfim_quantum" -> tfimQuantumEscalation();
            case "hubbard_fallback" -> hubbardFallback();
            default -> null;
        };
    }
}
