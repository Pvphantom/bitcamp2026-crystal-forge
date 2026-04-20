package io.crystalforge.minecraft;

import com.google.gson.JsonObject;
import net.minecraft.util.math.BlockPos;

public final class CrystalForgePayloadStore {
    private static JsonObject latestPayload;
    private static BlockPos sceneOrigin;

    private CrystalForgePayloadStore() {
    }

    public static synchronized void setLatestPayload(JsonObject payload) {
        latestPayload = payload;
        try {
            JsonObject scene = payload.getAsJsonObject("scene");
            JsonObject origin = scene.getAsJsonObject("origin");
            sceneOrigin = new BlockPos(
                origin.get("x").getAsInt(),
                origin.get("y").getAsInt(),
                origin.get("z").getAsInt()
            );
        } catch (Exception e) {
            CrystalForgeMod.LOGGER.warn("Crystal Forge: could not parse scene origin from payload, keeping previous");
        }
    }

    public static synchronized JsonObject getLatestPayload() {
        return latestPayload;
    }

    public static synchronized BlockPos getSceneOrigin() {
        return sceneOrigin;
    }
}
