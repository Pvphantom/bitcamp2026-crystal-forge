package io.crystalforge.minecraft;

import net.fabricmc.api.ClientModInitializer;

public final class CrystalForgeClientMod implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        CrystalForgeMod.LOGGER.info("Crystal Forge client hooks ready.");
    }
}
