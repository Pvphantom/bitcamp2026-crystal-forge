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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

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
        renderControlWall(world, origin, payload);
        renderWorkflowWall(world, origin, payload);
        renderObservablesWall(world, origin, payload);
        renderVqeChamber(world, origin, payload);
        renderQProbeRing(world, origin, payload);

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
                            ? Blocks.LIGHT_GRAY_STAINED_GLASS
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
            setBlock(world, origin.add(x, ROOM_HEIGHT - 1, -ROOM_HALF_DEPTH + 1), Blocks.OCHRE_FROGLIGHT);
            setBlock(world, origin.add(x, ROOM_HEIGHT - 1, ROOM_HALF_DEPTH - 1), Blocks.OCHRE_FROGLIGHT);
        }

        for (int y = 1; y <= 5; y++) {
            for (int z = -6; z <= 6; z++) {
                setBlock(world, origin.add(-13, y, z), Blocks.WHITE_CONCRETE);
                setBlock(world, origin.add(13, y, z), Blocks.WHITE_CONCRETE);
            }
        }

        for (int x = -10; x <= 10; x++) {
            for (int y = 1; y <= 5; y++) {
                setBlock(world, origin.add(x, y, -11), Blocks.WHITE_CONCRETE);
            }
        }

        for (int z = -6; z <= 6; z++) {
            setBlock(world, origin.add(-12, 1, z), Blocks.ORANGE_CONCRETE);
            setBlock(world, origin.add(12, 1, z), Blocks.LIME_CONCRETE);
        }
        for (int x = -10; x <= 10; x++) {
            setBlock(world, origin.add(x, 1, -10), Blocks.LIGHT_BLUE_CONCRETE);
        }
        for (int x = -4; x <= 4; x++) {
            setBlock(world, origin.add(x, 1, 10), Blocks.CYAN_CONCRETE);
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

    private static void renderControlWall(ServerWorld world, BlockPos origin, JsonObject payload) {
        JsonObject problem = payload.getAsJsonObject("problem");
        String family = problem.get("model_family").getAsString();
        JsonObject inputs = problem.getAsJsonObject("inputs");
        JsonObject parameters = inputs.getAsJsonObject("parameters");

        int wallX = origin.getX() - 13;
        for (int y = origin.getY() + 2; y <= origin.getY() + 5; y++) {
            setBlock(world, new BlockPos(wallX, y, origin.getZ() - 6), Blocks.POLISHED_BASALT);
            setBlock(world, new BlockPos(wallX, y, origin.getZ() + 6), Blocks.POLISHED_BASALT);
        }
        for (int z = -6; z <= 6; z++) {
            setBlock(world, new BlockPos(wallX, origin.getY() + 5, origin.getZ() + z), Blocks.POLISHED_BASALT);
        }

        Block familyBlock = "tfim".equals(family) ? Blocks.BLUE_CONCRETE : Blocks.PURPLE_CONCRETE;
        setBlock(world, new BlockPos(wallX, origin.getY() + 2, origin.getZ() - 5), Blocks.BLUE_CONCRETE);
        setBlock(world, new BlockPos(wallX, origin.getY() + 2, origin.getZ() - 3), Blocks.PURPLE_CONCRETE);
        setBlock(world, new BlockPos(wallX, origin.getY() + 2, origin.getZ() + 5), Blocks.LIME_CONCRETE);

        String[] keys = "hubbard".equals(family)
            ? new String[]{"t", "U", "mu"}
            : new String[]{"J", "h", "g"};
        int[] rows = {-1, 1, 3};
        for (int i = 0; i < keys.length; i++) {
            String key = keys[i];
            int rowZ = rows[i];
            setBlock(world, new BlockPos(wallX, origin.getY() + 2, rowZ), Blocks.ORANGE_CONCRETE);
            setBlock(world, new BlockPos(wallX, origin.getY() + 3, rowZ), familyBlock);
            setBlock(world, new BlockPos(wallX, origin.getY() + 4, rowZ), Blocks.LIME_CONCRETE);

            double value = parameters.has(key) ? parameters.get(key).getAsDouble() : 0.0;
            int lights = Math.max(1, Math.min(4, (int) Math.ceil(Math.abs(value))));
            for (int n = 0; n < lights; n++) {
                setBlock(world, new BlockPos(wallX - 1 - n, origin.getY() + 3, rowZ), Blocks.GLOWSTONE);
            }
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

    private static void renderWorkflowWall(ServerWorld world, BlockPos origin, JsonObject payload) {
        JsonObject trust = payload.getAsJsonObject("trust");
        JsonObject routing = payload.getAsJsonObject("routing");
        JsonObject workflow = payload.getAsJsonObject("workflow");
        String risk = trust.get("risk_label").getAsString();
        String pathType = workflow.get("active_path_type").getAsString();
        String route = workflow.get("route_label").isJsonNull() ? "none" : workflow.get("route_label").getAsString();

        Block indicator = switch (risk) {
            case "safe" -> Blocks.PEARLESCENT_FROGLIGHT;
            case "warning" -> Blocks.GLOWSTONE;
            default -> Blocks.OCHRE_FROGLIGHT;
        };

        Block pathBlock = switch (pathType) {
            case "quantum" -> Blocks.SEA_LANTERN;
            case "exact_fallback" -> Blocks.SMOOTH_STONE;
            default -> Blocks.GRAY_CONCRETE;
        };
        Block routeBlock = switch (route) {
            case "mean_field" -> Blocks.LIME_CONCRETE;
            case "quantum_frontier" -> Blocks.CYAN_CONCRETE;
            case "scalable_classical" -> Blocks.WHITE_CONCRETE;
            default -> Blocks.ORANGE_CONCRETE;
        };
        Block routingAvailable = routing.get("available").getAsBoolean() ? Blocks.PEARLESCENT_FROGLIGHT : Blocks.GRAY_CONCRETE;

        int wallX = origin.getX() + 13;
        for (int y = origin.getY() + 1; y <= origin.getY() + 5; y++) {
            setBlock(world, new BlockPos(wallX, y, origin.getZ() - 6), Blocks.POLISHED_BASALT);
            setBlock(world, new BlockPos(wallX, y, origin.getZ() + 6), Blocks.POLISHED_BASALT);
        }
        for (int z = -6; z <= 6; z++) {
            setBlock(world, new BlockPos(wallX, origin.getY() + 5, origin.getZ() + z), Blocks.POLISHED_BASALT);
        }

        for (int y = 0; y < 4; y++) {
            setBlock(world, new BlockPos(wallX, origin.getY() + 2 + y, origin.getZ() - 4), indicator);
            setBlock(world, new BlockPos(wallX, origin.getY() + 2 + y, origin.getZ() - 1), routeBlock);
            setBlock(world, new BlockPos(wallX, origin.getY() + 2 + y, origin.getZ() + 2), pathBlock);
        }
        setBlock(world, new BlockPos(wallX, origin.getY() + 2, origin.getZ() + 5), routingAvailable);
        setBlock(world, new BlockPos(wallX, origin.getY() + 3, origin.getZ() + 5), routingAvailable);

        double riskValue = trust.get("max_abs_error").getAsDouble();
        int riskHeight = Math.max(1, Math.min(4, (int) Math.ceil(riskValue * 4.0)));
        for (int i = 0; i < riskHeight; i++) {
            setBlock(world, new BlockPos(wallX - 1, origin.getY() + 1 + i, origin.getZ() - 6), Blocks.OCHRE_FROGLIGHT);
        }
    }

    private static void renderObservablesWall(ServerWorld world, BlockPos origin, JsonObject payload) {
        JsonObject observables = payload.getAsJsonObject("observables").getAsJsonObject("global");
        List<Map.Entry<String, JsonElement>> entries = new ArrayList<>(observables.entrySet());
        entries.sort(Comparator.comparing(Map.Entry::getKey));

        int startX = origin.getX() - 9;
        int z = origin.getZ() - 11;
        int column = 0;
        for (Map.Entry<String, JsonElement> entry : entries) {
            double value = entry.getValue().getAsDouble();
            int height = Math.max(1, Math.min(6, (int) Math.ceil(Math.abs(value) * 5.0)));
            Block block = observableBlock(entry.getKey(), value);
            int x = startX + column * 3;
            setBlock(world, new BlockPos(x, origin.getY() + 1, z), Blocks.POLISHED_BASALT);
            for (int i = 1; i <= height; i++) {
                setBlock(world, new BlockPos(x, origin.getY() + 1 + i, z), block);
            }
            setBlock(world, new BlockPos(x, origin.getY() + 2 + height, z), Blocks.END_ROD);
            column += 1;
        }
    }

    private static void renderVqeChamber(ServerWorld world, BlockPos origin, JsonObject payload) {
        JsonObject workflow = payload.getAsJsonObject("workflow");
        JsonObject solvers = payload.getAsJsonObject("solvers");
        boolean quantum = "quantum".equals(workflow.get("active_path_type").getAsString());
        JsonObject strongSolver = solvers.getAsJsonObject("strong_solver");

        int baseZ = origin.getZ() + 13;
        for (int dx = -3; dx <= 3; dx++) {
            for (int dz = 0; dz <= 3; dz++) {
                setBlock(world, new BlockPos(origin.getX() + dx, origin.getY() + 1, baseZ + dz), Blocks.DEEPSLATE_TILES);
            }
        }
        for (int dx = -3; dx <= 3; dx++) {
            for (int dy = 2; dy <= 5; dy++) {
                if (Math.abs(dx) == 3) {
                    setBlock(world, new BlockPos(origin.getX() + dx, origin.getY() + dy, baseZ), Blocks.LIGHT_BLUE_STAINED_GLASS);
                    setBlock(world, new BlockPos(origin.getX() + dx, origin.getY() + dy, baseZ + 3), Blocks.LIGHT_BLUE_STAINED_GLASS);
                }
            }
        }

        Block shell = quantum ? Blocks.CYAN_CONCRETE : Blocks.GRAY_CONCRETE;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = 2; dy <= 4; dy++) {
                setBlock(world, new BlockPos(origin.getX() + dx, origin.getY() + dy, baseZ + 1), shell);
                setBlock(world, new BlockPos(origin.getX() + dx, origin.getY() + dy, baseZ + 2), shell);
            }
        }
        if (quantum) {
            setBlock(world, new BlockPos(origin.getX(), origin.getY() + 3, baseZ + 1), Blocks.PEARLESCENT_FROGLIGHT);
            setBlock(world, new BlockPos(origin.getX(), origin.getY() + 3, baseZ + 2), Blocks.PEARLESCENT_FROGLIGHT);
        } else {
            setBlock(world, new BlockPos(origin.getX(), origin.getY() + 3, baseZ + 1), Blocks.GLOWSTONE);
            setBlock(world, new BlockPos(origin.getX(), origin.getY() + 3, baseZ + 2), Blocks.GLOWSTONE);
        }

        if (strongSolver.get("available").getAsBoolean()) {
            JsonObject metadata = strongSolver.getAsJsonObject("metadata");
            if (metadata.has("parameter_count")) {
                int count = Math.min(5, Math.max(1, metadata.get("parameter_count").getAsInt() / 2));
                for (int i = 0; i < count; i++) {
                    setBlock(world, new BlockPos(origin.getX() - 3 + i, origin.getY() + 2, baseZ + 4), Blocks.END_ROD);
                }
            }
        }
    }

    private static void renderQProbeRing(ServerWorld world, BlockPos origin, JsonObject payload) {
        JsonObject measurement = payload.getAsJsonObject("measurement");
        if (!measurement.get("enabled").getAsBoolean()) {
            return;
        }

        JsonObject qprobe = measurement.getAsJsonObject("qprobe");
        JsonObject adaptive = measurement.getAsJsonObject("adaptive_qprobe");
        int[] xs = {-5, -2, 2, 5, 2, -2};
        int[] zs = {0, -4, -4, 0, 4, 4};
        int centerY = origin.getY() + 7;

        int activeNodes = 0;
        if (qprobe != null && qprobe.has("recommended_groups") && !qprobe.get("recommended_groups").isJsonNull()) {
            activeNodes = qprobe.getAsJsonArray("recommended_groups").size();
        }
        int adaptiveSteps = 0;
        if (adaptive != null && adaptive.has("steps") && !adaptive.get("steps").isJsonNull()) {
            adaptiveSteps = adaptive.getAsJsonArray("steps").size();
        }

        for (int i = 0; i < xs.length; i++) {
            BlockPos nodePos = new BlockPos(origin.getX() + xs[i], centerY, origin.getZ() + zs[i]);
            boolean active = i < activeNodes;
            setBlock(world, nodePos, active ? Blocks.CYAN_CONCRETE : Blocks.GRAY_CONCRETE);
            if (active) {
                setBlock(world, nodePos.up(), Blocks.OCHRE_FROGLIGHT);
            }
            if (i < adaptiveSteps) {
                setBlock(world, nodePos.up(2), Blocks.END_ROD);
            }
        }

        for (int i = 0; i < xs.length; i++) {
            int next = (i + 1) % xs.length;
            drawSimpleLine(
                world,
                new BlockPos(origin.getX() + xs[i], centerY, origin.getZ() + zs[i]),
                new BlockPos(origin.getX() + xs[next], centerY, origin.getZ() + zs[next]),
                Blocks.LIGHT_GRAY_STAINED_GLASS
            );
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

    private static Block observableBlock(String key, double value) {
        if ("energy".equals(key)) {
            return Blocks.CYAN_STAINED_GLASS;
        }
        if (key.contains("M") || key.contains("Z")) {
            return value >= 0.0 ? Blocks.RED_CONCRETE : Blocks.BLUE_CONCRETE;
        }
        if (key.contains("Pair")) {
            return Blocks.PURPLE_CONCRETE;
        }
        return Blocks.LIGHT_BLUE_CONCRETE;
    }

    private static void drawSimpleLine(ServerWorld world, BlockPos a, BlockPos b, Block block) {
        if (a.getY() != b.getY()) {
            return;
        }
        if (a.getX() == b.getX()) {
            int zMin = Math.min(a.getZ(), b.getZ());
            int zMax = Math.max(a.getZ(), b.getZ());
            for (int z = zMin + 1; z < zMax; z++) {
                setBlock(world, new BlockPos(a.getX(), a.getY(), z), block);
            }
            return;
        }
        if (a.getZ() == b.getZ()) {
            int xMin = Math.min(a.getX(), b.getX());
            int xMax = Math.max(a.getX(), b.getX());
            for (int x = xMin + 1; x < xMax; x++) {
                setBlock(world, new BlockPos(x, a.getY(), a.getZ()), block);
            }
        }
    }
}
