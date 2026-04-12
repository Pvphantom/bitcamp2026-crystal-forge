package io.crystalforge.minecraft;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.text.Text;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public final class CrystalForgeBridge {
    private static final HttpClient HTTP_CLIENT = HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_1_1)
        .connectTimeout(Duration.ofSeconds(3))
        .build();

    private CrystalForgeBridge() {
    }

    public static boolean refresh(CrystalForgeWorkflowRequest request) {
        HttpRequest httpRequest = HttpRequest.newBuilder()
            .uri(URI.create(CrystalForgeConfig.backendUrl()))
            .header("Content-Type", "application/json")
            .timeout(Duration.ofSeconds(20))
            .POST(HttpRequest.BodyPublishers.ofString(request.jsonBody()))
            .build();

        try {
            HttpResponse<String> response = HTTP_CLIENT.send(httpRequest, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                CrystalForgeMod.LOGGER.error("Crystal Forge backend returned status {}: {}", response.statusCode(), response.body());
                return false;
            }
            JsonObject payload = JsonParser.parseString(response.body()).getAsJsonObject();
            CrystalForgePayloadStore.setLatestPayload(payload);
            CrystalForgeMod.LOGGER.info("Crystal Forge payload updated: update_id={}", payload.get("update_id").getAsInt());
            return true;
        } catch (IOException | InterruptedException ex) {
            CrystalForgeMod.LOGGER.error("Crystal Forge backend fetch failed", ex);
            if (ex instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            return false;
        }
    }

    public static void renderLatestSummary(ServerCommandSource source) {
        JsonObject payload = CrystalForgePayloadStore.getLatestPayload();
        if (payload == null) {
            source.sendError(Text.literal("Crystal Forge: no payload loaded yet. Run /crystalforge refresh first."));
            return;
        }

        JsonObject problem = payload.getAsJsonObject("problem");
        JsonObject workflow = payload.getAsJsonObject("workflow");
        JsonObject trust = payload.getAsJsonObject("trust");
        JsonObject observables = payload.getAsJsonObject("observables");
        JsonObject global = observables.getAsJsonObject("global");

        String modelFamily = problem.get("model_family").getAsString();
        String activeSolver = workflow.get("active_solver").getAsString();
        String pathType = workflow.get("active_path_type").getAsString();
        String risk = trust.get("risk_label").getAsString();

        source.sendFeedback(() -> Text.literal("Crystal Forge model: " + modelFamily), false);
        source.sendFeedback(() -> Text.literal("Active solver: " + activeSolver + " | path: " + pathType), false);
        source.sendFeedback(() -> Text.literal("CorrMap risk: " + risk), false);
        source.sendFeedback(() -> Text.literal("Energy: " + global.get("energy").getAsDouble()), false);

        CrystalForgeRenderer.renderIntoWorld(source, payload);
    }
}
