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
                    yield refreshAndRender(serverPlayer, "Switched model to TFIM.");
                }
                case MODEL_HUBBARD -> {
                    CrystalForgeSession.resetToModel("hubbard");
                    yield refreshAndRender(serverPlayer, "Switched model to Hubbard.");
                }
                case RUN -> refreshAndRender(serverPlayer, "Re-running current workflow.");
                case PARAM_DELTA -> {
                    CrystalForgeSession.adjustParameter(action.parameterKey(), action.delta());
                    double current = CrystalForgeSession.activeParameterValue(action.parameterKey());
                    yield refreshAndRender(serverPlayer, action.parameterKey() + " -> " + current);
                }
            };

            return ok ? ActionResult.SUCCESS : ActionResult.FAIL;
        });
    }

    private static boolean refreshAndRender(ServerPlayerEntity player, String message) {
        player.sendMessage(Text.literal("Crystal Forge: " + message), false);
        boolean ok = CrystalForgeBridge.refresh(CrystalForgeSession.currentRequest());
        if (!ok) {
            player.sendMessage(Text.literal("Crystal Forge: backend fetch failed. Check runClient and backend logs."), false);
            return false;
        }
        CrystalForgeBridge.renderLatestSummary(player.getCommandSource());
        return true;
    }

    private static ControlAction actionAt(BlockPos pos) {
        int x = pos.getX();
        int y = pos.getY();
        int z = pos.getZ();

        if (x != -13) {
            return null;
        }

        if (y == 2 && z == -5) {
            return new ControlAction(ControlType.MODEL_TFIM, null, 0.0);
        }
        if (y == 2 && z == -3) {
            return new ControlAction(ControlType.MODEL_HUBBARD, null, 0.0);
        }
        if (y == 2 && z == 5) {
            return new ControlAction(ControlType.RUN, null, 0.0);
        }

        String[] params = CrystalForgeSession.activeParameterOrder();
        int[] rows = {-1, 1, 3};
        for (int i = 0; i < rows.length; i++) {
            if (z != rows[i]) {
                continue;
            }
            String key = params[i];
            if (y == 2) {
                return new ControlAction(ControlType.PARAM_DELTA, key, -CrystalForgeSession.stepSize(key));
            }
            if (y == 4) {
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
