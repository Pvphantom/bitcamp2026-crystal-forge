package io.crystalforge.minecraft;

public final class CrystalForgeConfig {
    private static final String BACKEND_URL_PROPERTY = "crystalforge.backendUrl";
    private static final String BACKEND_URL_ENV = "CRYSTALFORGE_BACKEND_URL";

    private CrystalForgeConfig() {
    }

    public static final String DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/api/minecraft/workflow";

    public static String backendUrl() {
        String fromProperty = System.getProperty(BACKEND_URL_PROPERTY);
        if (fromProperty != null && !fromProperty.isBlank()) {
            return fromProperty;
        }
        String fromEnv = System.getenv(BACKEND_URL_ENV);
        if (fromEnv != null && !fromEnv.isBlank()) {
            return fromEnv;
        }
        return DEFAULT_BACKEND_URL;
    }
}
