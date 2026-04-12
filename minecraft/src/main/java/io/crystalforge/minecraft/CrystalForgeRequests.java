package io.crystalforge.minecraft;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

public final class CrystalForgeRequests {
    private CrystalForgeRequests() {
    }

    public static CrystalForgeWorkflowRequest tfimQuantumEscalation() {
        JsonObject root = baseRequest("tfim", 2, 2);
        JsonObject parameters = new JsonObject();
        parameters.addProperty("J", 1.0);
        parameters.addProperty("h", 0.8);
        parameters.addProperty("g", 0.0);
        root.add("parameters", parameters);
        root.add("qprobe_targets", targetArray("Mz", "ZZ_nn", "Mstag2"));
        root.addProperty("qprobe_tolerance", 0.03);
        root.addProperty("qprobe_shots_per_group", 4000);
        root.addProperty("qprobe_readout_flip_prob", 0.02);
        root.addProperty("qprobe_seed", 7);
        return new CrystalForgeWorkflowRequest(root);
    }

    public static CrystalForgeWorkflowRequest tfimSafe() {
        JsonObject root = baseRequest("tfim", 2, 2);
        JsonObject parameters = new JsonObject();
        parameters.addProperty("J", 1.0);
        parameters.addProperty("h", 0.1);
        parameters.addProperty("g", 1.4);
        root.add("parameters", parameters);
        root.add("qprobe_targets", targetArray("Mz", "ZZ_nn", "Mstag2"));
        root.addProperty("qprobe_tolerance", 0.03);
        root.addProperty("qprobe_shots_per_group", 4000);
        root.addProperty("qprobe_readout_flip_prob", 0.02);
        root.addProperty("qprobe_seed", 7);
        return new CrystalForgeWorkflowRequest(root);
    }

    public static CrystalForgeWorkflowRequest hubbardFallback() {
        JsonObject root = baseRequest("hubbard", 2, 2);
        JsonObject parameters = new JsonObject();
        parameters.addProperty("t", 1.0);
        parameters.addProperty("U", 4.0);
        parameters.addProperty("mu", 2.0);
        root.add("parameters", parameters);
        root.add("qprobe_targets", targetArray("D", "Ms2", "Cs_max"));
        root.addProperty("qprobe_tolerance", 0.03);
        root.addProperty("qprobe_shots_per_group", 2000);
        root.addProperty("qprobe_readout_flip_prob", 0.01);
        root.addProperty("qprobe_seed", 7);
        return new CrystalForgeWorkflowRequest(root);
    }

    public static CrystalForgeWorkflowRequest fromPreset(String name) {
        return switch (name) {
            case "tfim_safe" -> tfimSafe();
            case "tfim_quantum" -> tfimQuantumEscalation();
            case "hubbard_fallback" -> hubbardFallback();
            default -> null;
        };
    }

    private static JsonObject baseRequest(String modelFamily, int lx, int ly) {
        JsonObject root = new JsonObject();
        root.addProperty("model_family", modelFamily);
        root.addProperty("Lx", lx);
        root.addProperty("Ly", ly);
        return root;
    }

    private static JsonArray targetArray(String... targets) {
        JsonArray array = new JsonArray();
        for (String target : targets) {
            array.add(target);
        }
        return array;
    }
}
