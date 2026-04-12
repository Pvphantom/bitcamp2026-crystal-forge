package io.crystalforge.minecraft;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;

public final class CrystalForgeRenderer {
    private static final int FLOOR_Y = 64;
    private static final int PLATFORM_Y = 65;
    private static final int ROOM_HALF_WIDTH = 17;
    private static final int ROOM_HALF_DEPTH = 12;
    private static final int ROOM_HEIGHT = 8;

    private CrystalForgeRenderer() {
    }

    public static void renderIntoWorld(ServerCommandSource source, JsonObject payload) {
        ServerWorld world = source.getWorld();
        BlockPos origin = originFromPayload(payload);

        clearRenderRegion(world, origin);
        buildRoomShell(world, origin);
        buildPlatform(world, origin);
        renderSitesAndBonds(world, origin, payload);
        renderWorkflowIndicator(world, origin, payload);

        CrystalForgeMod.LOGGER.info(
            "Crystal Forge world render complete at origin ({}, {}, {})",
            origin.getX(),
            origin.getY(),
            origin.getZ()
        );
    }

    private static BlockPos originFromPayload(JsonObject payload) {
        JsonObject scene = payload.getAsJsonObject("scene");
        JsonObject origin = scene.getAsJsonObject("origin");
        return new BlockPos(
            origin.get("x").getAsInt(),
            origin.get("y").getAsInt(),
            origin.get("z").getAsInt()
        );
    }

    private static void clearRenderRegion(ServerWorld world, BlockPos origin) {
        for (int x = -ROOM_HALF_WIDTH; x <= ROOM_HALF_WIDTH; x++) {
            for (int z = -ROOM_HALF_DEPTH; z <= ROOM_HALF_DEPTH + 8; z++) {
                for (int y = 1; y <= ROOM_HEIGHT + 2; y++) {
                    setBlock(world, origin.add(x, y, z), Blocks.AIR);
                }
            }
        }
    }

    private static void buildRoomShell(ServerWorld world, BlockPos origin) {
        for (int x = -ROOM_HALF_WIDTH; x <= ROOM_HALF_WIDTH; x++) {
            for (int z = -ROOM_HALF_DEPTH; z <= ROOM_HALF_DEPTH; z++) {
                boolean border = Math.abs(x) == ROOM_HALF_WIDTH || Math.abs(z) == ROOM_HALF_DEPTH;
                Block floor = border ? Blocks.DEEPSLATE_TILES : Blocks.POLISHED_DEEPSLATE;
                setBlock(world, origin.add(x, 0, z), floor);
                if (border) {
                    for (int y = 1; y <= ROOM_HEIGHT; y++) {
                        Block wall = (Math.abs(x) == ROOM_HALF_WIDTH && Math.abs(z) <= 6 && y >= 2 && y <= 5)
                            || (Math.abs(z) == ROOM_HALF_DEPTH && Math.abs(x) <= 10 && y >= 2 && y <= 5)
                            ? Blocks.TINTED_GLASS
                            : Blocks.DEEPSLATE_TILES;
                        setBlock(world, origin.add(x, y, z), wall);
                    }
                }
            }
        }

        for (int x = -ROOM_HALF_WIDTH; x <= ROOM_HALF_WIDTH; x++) {
            for (int z = -ROOM_HALF_DEPTH; z <= ROOM_HALF_DEPTH; z++) {
                if (Math.abs(x) == ROOM_HALF_WIDTH || Math.abs(z) == ROOM_HALF_DEPTH) {
                    continue;
                }
                setBlock(world, origin.add(x, ROOM_HEIGHT, z), Blocks.DEEPSLATE_TILES);
            }
        }

        for (int x = -ROOM_HALF_WIDTH + 2; x <= ROOM_HALF_WIDTH - 2; x += 7) {
            setBlock(world, origin.add(x, ROOM_HEIGHT - 1, -ROOM_HALF_DEPTH + 1), Blocks.SEA_LANTERN);
            setBlock(world, origin.add(x, ROOM_HEIGHT - 1, ROOM_HALF_DEPTH - 1), Blocks.SEA_LANTERN);
        }
    }

    private static void buildPlatform(ServerWorld world, BlockPos origin) {
        for (int x = -4; x <= 4; x++) {
            for (int z = -4; z <= 4; z++) {
                boolean edge = Math.abs(x) == 4 || Math.abs(z) == 4;
                Block block = edge ? Blocks.IRON_BLOCK : Blocks.POLISHED_DEEPSLATE;
                setBlock(world, origin.add(x, PLATFORM_Y - FLOOR_Y, z), block);
            }
        }
        setBlock(world, origin.add(-2, PLATFORM_Y - FLOOR_Y, -2), Blocks.SEA_LANTERN);
        setBlock(world, origin.add(2, PLATFORM_Y - FLOOR_Y, -2), Blocks.SEA_LANTERN);
        setBlock(world, origin.add(-2, PLATFORM_Y - FLOOR_Y, 2), Blocks.SEA_LANTERN);
        setBlock(world, origin.add(2, PLATFORM_Y - FLOOR_Y, 2), Blocks.SEA_LANTERN);
    }

    private static void renderSitesAndBonds(ServerWorld world, BlockPos origin, JsonObject payload) {
        JsonObject problem = payload.getAsJsonObject("problem");
        String modelFamily = problem.get("model_family").getAsString();
        JsonObject observables = payload.getAsJsonObject("observables");
        JsonArray sites = observables.getAsJsonArray("site_values");
        JsonArray bonds = observables.getAsJsonArray("bond_values");

        for (JsonElement element : sites) {
            JsonObject site = element.getAsJsonObject();
            BlockPos sitePos = sitePosition(origin, site.get("lattice_x").getAsInt(), site.get("lattice_y").getAsInt());
            if ("tfim".equals(modelFamily)) {
                renderTfimSite(world, sitePos, site);
            } else {
                renderHubbardSite(world, sitePos, site);
            }
        }

        for (JsonElement element : bonds) {
            JsonObject bond = element.getAsJsonObject();
            BlockPos a = sitePositionFromId(origin, sites, bond.get("i").getAsInt());
            BlockPos b = sitePositionFromId(origin, sites, bond.get("j").getAsInt());
            renderBond(world, a, b, bond);
        }
    }

    private static BlockPos sitePositionFromId(BlockPos origin, JsonArray sites, int siteId) {
        for (JsonElement element : sites) {
            JsonObject site = element.getAsJsonObject();
            if (site.get("site_id").getAsInt() == siteId) {
                return sitePosition(origin, site.get("lattice_x").getAsInt(), site.get("lattice_y").getAsInt());
            }
        }
        return origin.up(2);
    }

    private static BlockPos sitePosition(BlockPos origin, int latticeX, int latticeY) {
        int x = -2 + latticeX * 4;
        int z = -2 + latticeY * 4;
        return origin.add(x, 2, z);
    }

    private static void renderHubbardSite(ServerWorld world, BlockPos sitePos, JsonObject site) {
        JsonObject render = site.getAsJsonObject("render");
        String occupancy = render.get("occupancy_label").getAsString();
        setBlock(world, sitePos, Blocks.POLISHED_BASALT);
        setBlock(world, sitePos.up(), Blocks.GRAY_CONCRETE);
        setBlock(world, sitePos.up(2), hubbardOccupancyBlock(occupancy));
    }

    private static void renderTfimSite(ServerWorld world, BlockPos sitePos, JsonObject site) {
        JsonObject render = site.getAsJsonObject("render");
        int sign = render.get("primary_sign").getAsInt();
        double magnitude = render.get("primary_magnitude").getAsDouble();
        double secondary = render.has("secondary_magnitude") && !render.get("secondary_magnitude").isJsonNull()
            ? render.get("secondary_magnitude").getAsDouble()
            : 0.0;
        int height = Math.max(1, (int) Math.ceil(magnitude * 4.0));

        setBlock(world, sitePos, Blocks.POLISHED_BASALT);
        Block pillar = sign >= 0 ? Blocks.RED_CONCRETE : Blocks.BLUE_CONCRETE;
        for (int i = 1; i <= height; i++) {
            setBlock(world, sitePos.up(i), pillar);
        }
        if (secondary > 0.15) {
            setBlock(world, sitePos.east(), Blocks.LIGHT_BLUE_STAINED_GLASS);
        }
    }

    private static void renderBond(ServerWorld world, BlockPos a, BlockPos b, JsonObject bond) {
        JsonObject render = bond.getAsJsonObject("render");
        double magnitude = render.get("magnitude").getAsDouble();
        Block bridge = magnitude > 0.5 ? Blocks.CYAN_STAINED_GLASS : Blocks.LIGHT_GRAY_CONCRETE;

        if (a.getX() == b.getX()) {
            int zMin = Math.min(a.getZ(), b.getZ());
            int zMax = Math.max(a.getZ(), b.getZ());
            for (int z = zMin + 1; z < zMax; z++) {
                setBlock(world, new BlockPos(a.getX(), PLATFORM_Y + 2, z), bridge);
            }
        } else if (a.getZ() == b.getZ()) {
            int xMin = Math.min(a.getX(), b.getX());
            int xMax = Math.max(a.getX(), b.getX());
            for (int x = xMin + 1; x < xMax; x++) {
                setBlock(world, new BlockPos(x, PLATFORM_Y + 2, a.getZ()), bridge);
            }
        }
    }

    private static void renderWorkflowIndicator(ServerWorld world, BlockPos origin, JsonObject payload) {
        JsonObject trust = payload.getAsJsonObject("trust");
        JsonObject workflow = payload.getAsJsonObject("workflow");
        String risk = trust.get("risk_label").getAsString();
        String pathType = workflow.get("active_path_type").getAsString();

        Block indicator = switch (risk) {
            case "safe" -> Blocks.LIME_CONCRETE;
            case "warning" -> Blocks.YELLOW_CONCRETE;
            default -> Blocks.RED_CONCRETE;
        };

        Block chamber = switch (pathType) {
            case "quantum" -> Blocks.CYAN_STAINED_GLASS;
            case "exact_fallback" -> Blocks.WHITE_CONCRETE;
            default -> Blocks.GRAY_CONCRETE;
        };

        int x = origin.getX() + 12;
        int z = origin.getZ();
        for (int y = origin.getY() + 2; y <= origin.getY() + 4; y++) {
            setBlock(world, new BlockPos(x, y, z), indicator);
        }

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = 0; dy <= 2; dy++) {
                setBlock(world, new BlockPos(origin.getX() + dx, origin.getY() + 2 + dy, origin.getZ() + 13), chamber);
            }
        }
        if ("quantum".equals(pathType)) {
            setBlock(world, new BlockPos(origin.getX(), origin.getY() + 3, origin.getZ() + 13), Blocks.SEA_LANTERN);
        }
    }

    private static Block hubbardOccupancyBlock(String occupancy) {
        return switch (occupancy) {
            case "up" -> Blocks.BLUE_CONCRETE;
            case "down" -> Blocks.RED_CONCRETE;
            case "double" -> Blocks.PURPLE_CONCRETE;
            default -> Blocks.GRAY_CONCRETE;
        };
    }

    private static void setBlock(ServerWorld world, BlockPos pos, Block block) {
        BlockState state = block.getDefaultState();
        world.setBlockState(pos, state, Block.NOTIFY_LISTENERS);
    }
}
