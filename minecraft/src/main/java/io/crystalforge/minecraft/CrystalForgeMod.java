package io.crystalforge.minecraft;

import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.minecraft.server.command.CommandManager;
import net.minecraft.text.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class CrystalForgeMod implements ModInitializer {
    public static final String MOD_ID = "crystalforge";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(
                CommandManager.literal("crystalforge")
                    .then(CommandManager.literal("refresh")
                        .executes(context -> {
                            context.getSource().sendFeedback(() -> Text.literal("Crystal Forge: fetching default TFIM workflow..."), false);
                            boolean ok = CrystalForgeBridge.refresh(CrystalForgeRequests.tfimQuantumEscalation());
                            if (!ok) {
                                context.getSource().sendError(Text.literal("Crystal Forge: backend fetch failed. Check the runClient terminal and backend logs."));
                                return 0;
                            }
                            CrystalForgeBridge.renderLatestSummary(context.getSource());
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
                                boolean ok = CrystalForgeBridge.refresh(request);
                                if (!ok) {
                                    context.getSource().sendError(Text.literal("Crystal Forge: backend fetch failed. Check the runClient terminal and backend logs."));
                                    return 0;
                                }
                                CrystalForgeBridge.renderLatestSummary(context.getSource());
                                return 1;
                            })))
                    .then(CommandManager.literal("status")
                        .executes(context -> {
                            CrystalForgeBridge.renderLatestSummary(context.getSource());
                            return 1;
                        }))
            )
        );

        LOGGER.info("Crystal Forge Minecraft bridge initialized.");
    }
}
