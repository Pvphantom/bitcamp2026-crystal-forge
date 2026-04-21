package io.crystalforge.minecraft;

import com.google.gson.JsonObject;

public final class CrystalForgeSession {
    private static CrystalForgeWorkflowRequest currentRequest = CrystalForgeRequests.tfimQuantumEscalation();

    private CrystalForgeSession() {
    }

    public static synchronized CrystalForgeWorkflowRequest currentRequest() {
        return cloneRequest(currentRequest);
    }

    public static synchronized void setCurrentRequest(CrystalForgeWorkflowRequest request) {
        currentRequest = cloneRequest(request);
    }

    public static synchronized String modelFamily() {
        return currentRequest.payload().get("model_family").getAsString();
    }

    public static synchronized void resetToModel(String modelFamily) {
        currentRequest = switch (modelFamily) {
            case "hubbard" -> CrystalForgeRequests.hubbardFallback();
            default -> CrystalForgeRequests.tfimQuantumEscalation();
        };
    }

    public static synchronized String[] activeParameterOrder() {
        return "hubbard".equals(modelFamily())
            ? new String[]{"t", "U", "mu"}
            : new String[]{"J", "h", "g"};
    }

    public static synchronized double activeParameterValue(String key) {
        JsonObject parameters = currentRequest.payload().getAsJsonObject("parameters");
        return parameters.has(key) ? parameters.get(key).getAsDouble() : 0.0;
    }

    public static synchronized void adjustParameter(String key, double delta) {
        JsonObject parameters = currentRequest.payload().getAsJsonObject("parameters");
        double value = parameters.has(key) ? parameters.get(key).getAsDouble() : 0.0;
        double next = value + delta;
        double min = switch (key) {
            case "U", "t", "J", "h" -> 0.0;
            default -> -4.0;
        };
        double max = switch (key) {
            case "U" -> 12.0;
            case "t", "J", "h" -> 4.0;
            default -> 4.0;
        };
        next = Math.max(min, Math.min(max, next));
        parameters.addProperty(key, roundToTenths(next));
    }

    public static double stepSize(String key) {
        return switch (key) {
            case "U" -> 0.5;
            case "mu", "g" -> 0.25;
            default -> 0.1;
        };
    }

    private static CrystalForgeWorkflowRequest cloneRequest(CrystalForgeWorkflowRequest request) {
        return new CrystalForgeWorkflowRequest(request.payload().deepCopy());
    }

    private static double roundToTenths(double value) {
        return Math.round(value * 100.0) / 100.0;
    }
}
