package io.crystalforge.minecraft;

import net.fabricmc.fabric.api.event.player.UseBlockCallback;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.ActionResult;
import net.minecraft.util.math.BlockPos;

public final class CrystalForgeControls {
    private CrystalForgeControls() {
    }

    public static void register() {
        UseBlockCallback.EVENT.register((player, world, hand, hitResult) -> {
            if (world.isClient() || !(player instanceof ServerPlayerEntity serverPlayer)) {
                return ActionResult.PASS;
            }

            BlockPos pos = hitResult.getBlockPos();
            ControlAction action = actionAt(pos);
            if (action == null) {
                return ActionResult.PASS;
            }

            boolean ok = switch (action.type()) {
                case MODEL_TFIM -> {
                    CrystalForgeSession.resetToModel("tfim");
                    yield refreshAndRenderAsync(serverPlayer, "Switched model to TFIM.");
                }
                case MODEL_HUBBARD -> {
                    CrystalForgeSession.resetToModel("hubbard");
                    yield refreshAndRenderAsync(serverPlayer, "Switched model to Hubbard.");
                }
                case RUN -> refreshAndRenderAsync(serverPlayer, "Re-running current workflow.");
                case PARAM_DELTA -> {
                    CrystalForgeSession.adjustParameter(action.parameterKey(), action.delta());
                    double current = CrystalForgeSession.activeParameterValue(action.parameterKey());
                    yield refreshAndRenderAsync(serverPlayer, action.parameterKey() + " -> " + current);
                }
            };

            return ok ? ActionResult.SUCCESS : ActionResult.FAIL;
        });
    }

    /**
     * Sends the feedback message immediately and dispatches the backend
     * call + render onto a background thread so the server tick is not
     * blocked by HTTP latency.
     */
    private static boolean refreshAndRenderAsync(ServerPlayerEntity player, String message) {
        player.sendMessage(Text.literal("Crystal Forge: " + message), false);
        CrystalForgeWorkflowRequest request = CrystalForgeSession.currentRequest();
        Thread.ofVirtual().name("cf-refresh").start(() -> {
            boolean ok = CrystalForgeBridge.refresh(request);
            // Schedule the render back on the server thread
            player.getServer().execute(() -> {
                if (!ok) {
                    player.sendMessage(
                        Text.literal("Crystal Forge: backend fetch failed. Check runClient and backend logs."),
                        false
                    );
                    return;
                }
                CrystalForgeBridge.renderLatestSummary(player.getCommandSource());
                player.sendMessage(Text.literal("Crystal Forge: updated."), false);
            });
        });
        return true;
    }

    /**
     * Maps a clicked block position to a control action.
     * Uses the stored scene origin so click detection works
     * regardless of where the room was generated.
     */
    private static ControlAction actionAt(BlockPos pos) {
        BlockPos origin = CrystalForgePayloadStore.getSceneOrigin();
        if (origin == null) {
            // No scene rendered yet — use legacy absolute coords as fallback
            return actionAtAbsolute(pos, 0, 0);
        }
        int relX = pos.getX() - origin.getX();
        int relY = pos.getY() - origin.getY();
        int relZ = pos.getZ() - origin.getZ();
        return actionAtRelative(relX, relY, relZ);
    }

    private static ControlAction actionAtAbsolute(BlockPos pos, int originX, int originZ) {
        return actionAtRelative(
            pos.getX() - originX,
            pos.getY() - 64, // default origin Y
            pos.getZ() - originZ
        );
    }

    private static ControlAction actionAtRelative(int relX, int relY, int relZ) {
        // Control wall is at relX == -13
        if (relX != -13) {
            return null;
        }

        // Model switch buttons
        if (relY == 2 && relZ == -5) {
            return new ControlAction(ControlType.MODEL_TFIM, null, 0.0);
        }
        if (relY == 2 && relZ == -3) {
            return new ControlAction(ControlType.MODEL_HUBBARD, null, 0.0);
        }
        // Run / refresh button
        if (relY == 2 && relZ == 5) {
            return new ControlAction(ControlType.RUN, null, 0.0);
        }

        // Parameter adjustment rows
        String[] params = CrystalForgeSession.activeParameterOrder();
        int[] rows = {-1, 1, 3};
        for (int i = 0; i < rows.length; i++) {
            if (relZ != rows[i]) {
                continue;
            }
            String key = params[i];
            if (relY == 2) {
                return new ControlAction(ControlType.PARAM_DELTA, key, -CrystalForgeSession.stepSize(key));
            }
            if (relY == 4) {
                return new ControlAction(ControlType.PARAM_DELTA, key, CrystalForgeSession.stepSize(key));
            }
        }
        return null;
    }

    private enum ControlType {
        MODEL_TFIM,
        MODEL_HUBBARD,
        PARAM_DELTA,
        RUN
    }

    private record ControlAction(ControlType type, String parameterKey, double delta) {
    }
}
