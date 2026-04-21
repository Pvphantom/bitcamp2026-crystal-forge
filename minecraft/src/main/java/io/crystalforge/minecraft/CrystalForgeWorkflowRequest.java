package io.crystalforge.minecraft;

import com.google.gson.JsonObject;

public record CrystalForgeWorkflowRequest(JsonObject payload) {
    public String jsonBody() {
        return payload.toString();
    }
}
