package io.crystalforge.minecraft;

import com.google.gson.JsonObject;

public final class CrystalForgePayloadStore {
    private static JsonObject latestPayload;

    private CrystalForgePayloadStore() {
    }

    public static synchronized void setLatestPayload(JsonObject payload) {
        latestPayload = payload;
    }

    public static synchronized JsonObject getLatestPayload() {
        return latestPayload;
    }
}
