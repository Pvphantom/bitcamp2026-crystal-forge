package io.crystalforge.minecraft;

import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;

public final class CrystalForgeMod implements ModInitializer {
    public static final String MOD_ID = "crystalforge";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    /** Ensures the initial backend fetch + render only happens once per server session. */
    private static final AtomicBoolean initialRenderDone = new AtomicBoolean(false);

    @Override
    public void onInitialize() {
        CrystalForgeControls.register();
        registerCommands();
        registerAutoInit();

        LOGGER.info("Crystal Forge Minecraft bridge initialized.");
    }

    /**
     * On the first player join, automatically fetch the default preset
     * from the backend and render the scene so the user never has to
     * type /crystalforge refresh manually.
     */
    private void registerAutoInit() {
        ServerPlayConnectionEvents.JOIN.register((handler, sender, server) -> {
            if (!initialRenderDone.compareAndSet(false, true)) {
                return; // already done
            }
            ServerPlayerEntity player = handler.getPlayer();
            player.sendMessage(Text.literal("Crystal Forge: initializing scene..."), false);
            CrystalForgeWorkflowRequest request = CrystalForgeSession.currentRequest();
            Thread.ofVirtual().name("cf-init").start(() -> {
                boolean ok = CrystalForgeBridge.refresh(request);
                server.execute(() -> {
                    if (!ok) {
                        player.sendMessage(
                            Text.literal("Crystal Forge: auto-init failed. Is the backend running? Try /crystalforge refresh"),
                            false
                        );
                        initialRenderDone.set(false); // allow retry on next join
                        return;
                    }
                    CrystalForgeBridge.renderLatestSummary(player.getCommandSource());
                    player.sendMessage(Text.literal("Crystal Forge: scene ready. Right-click the control wall to interact."), false);
                });
            });
        });
    }

    private void registerCommands() {
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(
                CommandManager.literal("crystalforge")
                    .then(CommandManager.literal("refresh")
                        .executes(context -> {
                            context.getSource().sendFeedback(() -> Text.literal("Crystal Forge: refreshing current workflow..."), false);
                            Thread.ofVirtual().name("cf-cmd-refresh").start(() -> {
                                boolean ok = CrystalForgeBridge.refresh(CrystalForgeSession.currentRequest());
                                context.getSource().getServer().execute(() -> {
                                    if (!ok) {
                                        context.getSource().sendError(Text.literal("Crystal Forge: backend fetch failed."));
                                        return;
                                    }
                                    CrystalForgeBridge.renderLatestSummary(context.getSource());
                                });
                            });
                            return 1;
                        }))
                    .then(CommandManager.literal("preset")
                        .then(CommandManager.argument("name", StringArgumentType.string())
                            .executes(context -> {
                                String preset = StringArgumentType.getString(context, "name");
                                CrystalForgeWorkflowRequest request = CrystalForgeRequests.fromPreset(preset);
                                if (request == null) {
                                    context.getSource().sendError(Text.literal("Unknown preset: " + preset));
                                    return 0;
                                }
                                context.getSource().sendFeedback(() -> Text.literal("Crystal Forge: fetching preset " + preset + "..."), false);
                                CrystalForgeSession.setCurrentRequest(request);
                                Thread.ofVirtual().name("cf-cmd-preset").start(() -> {
                                    boolean ok = CrystalForgeBridge.refresh(request);
                                    context.getSource().getServer().execute(() -> {
                                        if (!ok) {
                                            context.getSource().sendError(Text.literal("Crystal Forge: backend fetch failed."));
                                            return;
                                        }
                                        CrystalForgeBridge.renderLatestSummary(context.getSource());
                                    });
                                });
                                return 1;
                            })))
                    .then(CommandManager.literal("status")
                        .executes(context -> {
                            CrystalForgeBridge.renderLatestSummary(context.getSource());
                            return 1;
                        }))
            )
        );
    }
}
