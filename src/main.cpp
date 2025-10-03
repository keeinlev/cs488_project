#include "cs488.h"
CS488Window CS488;

 
// draw something in each frame
static void draw() {
    for (int j = 0; j < globalHeight; j++) {
        for (int i = 0; i < globalWidth; i++) {
            //FrameBuffer.pixel(i, j) = float3(PCG32::rand()); // noise
            FrameBuffer.pixel(i, j) = float3(0.5f * (cos((i + globalFrameCount) * 0.1f) + 1.0f)); // moving cosine
        }
    }
}
static void A0(int argc, const char* argv[]) {
    // set the function to be called in the main loop
    CS488.process = draw;
}



// setting up lighting
static PointLightSource light;
static PointLightSource light2;
static AreaLightSource arealight;
static TriangleMesh arealightmesh;
static AreaLightSource arealight1;
static TriangleMesh arealightmesh1;
static AreaLightSource arealight2;
static TriangleMesh arealightmesh2;
static AreaLightSource arealight3;
static TriangleMesh arealightmesh3;


static void setupLightSource() {
    arealight.mesh = &arealightmesh;
    arealight1.mesh = &arealightmesh1;
    arealight2.mesh = &arealightmesh2;
    arealight3.mesh = &arealightmesh3;
    arealight1.vertices[0] = float3(4.0f, 1.0f, 0.0f);
    arealight1.vertices[1] = float3(4.0f, 0.0f, 2.0f);
    arealight1.vertices[2] = float3(4.0f, 3.0f, 2.0f);
    arealight1.vertices[3] = float3(4.0f, 2.0f, 0.0f);
    arealight.vertices[0] = float3(-1, -0.5, -1);
    arealight.vertices[1] = float3(-1, -0.5, 1);
    arealight.vertices[2] = float3(1, -0.5, 1);
    arealight.vertices[3] = float3(1, -0.5, -1);
    arealight2.vertices[0] = float3(-0.1, 0.38, -0.1);
    arealight2.vertices[1] = float3(-0.1, 0.38, 0.1);
    arealight2.vertices[2] = float3(0.1, 0.38, 0.1);
    arealight2.vertices[3] = float3(0.1, 0.38, -0.1);
    arealight3.vertices[0] = float3(2.5f, -0.5f, -1.5f);
    arealight3.vertices[1] = float3(2.5f, -0.5f, -0.5f);
    arealight3.vertices[2] = float3(1.0f, 0.0f, -0.5f);
    arealight3.vertices[3] = float3(1.0f, 0.0f, -1.5f);
    arealight.intensity = float3(0.1f, 0.9f, 0.5f);
    arealight1.intensity = float3(0.5f, 0.1f, 0.3f);
    arealight2.intensity = float3(10.0f, 10.0f, 10.0f);
    arealight3.intensity = float3(5.0f, 5.0f, 5.0f);


    arealight.mesh->createSingleQuad(arealight.vertices);
    //globalScene.addObject(arealight.mesh);
    globalScene.addLight(&arealight);
    arealight1.mesh->createSingleQuad(arealight1.vertices);
    globalScene.addLight(&arealight1);
    arealight2.mesh->createSingleQuad(arealight2.vertices);
    globalScene.addLight(&arealight2);
    arealight3.mesh->createSingleQuad(arealight3.vertices);
    globalScene.addLight(&arealight3);
}



// ======== you probably don't need to modify below in A1 to A3 ========
// loading .obj file from the command line arguments
static TriangleMesh mesh;
static void setupScene(int argc, const char* argv[]) {
    if (argc > 1) {
        if (strcmp(argv[1], "-e") == 0 || (argc > 2 && strcmp(argv[2], "-e") == 0)) {
            if (argc == 2) {
                printf("Specify .obj file in the first command line argument. Example: CS488.exe -e cornellbox.obj\n");
                printf("Making a single triangle instead.\n");
                mesh.createSingleTriangle();
            }
            enableEnvironmentMapping = true;
            int err;
            globalScene.envMap.load("../../media/uffizi_probe.hdr", err);
            if (err) {
                printf("Trying to load ../media/uffizi_probe.hdr");
                globalScene.envMap.load("../media/uffizi_probe.hdr", err);
            }
        }
        bool objLoadSucceed = true;
        if (strcmp(argv[1], "-e") != 0) {
            objLoadSucceed = mesh.load(argv[1]);
            if (argc == 2) {
                printf("Specify '-e' flag to enable environment mapping. Example: CS488.exe -e cornellbox.obj\n");
            }
        } else if (argc > 2) {
            objLoadSucceed = mesh.load(argv[2]);
        }
        if (!objLoadSucceed) {
            printf("Invalid .obj file.\n");
            printf("Making a single triangle instead.\n");
            mesh.createSingleTriangle();
        }
    } else {
        printf("Specify .obj file in the command line arguments and/or the '-e' flag to enable environment mapping. Example: CS488.exe -e cornellbox.obj\n");
        printf("Making a single triangle instead.\n");
        mesh.createSingleTriangle();
    }
    globalScene.addObject(&mesh);
}

static void setup1(int argc, const char* argv[]) {
    // mipmap
    setupScene(argc, argv);
    globalRenderType = RENDER_RAYTRACE;
    light.position = float3(3.0f, 3.0f, 3.0f);
    light.wattage = float3(1000.0f, 1000.0f, 1000.0f);
    globalScene.addLight(&light);
}

static void setup2() {
    // collisions
    globalRenderType = RENDER_RAYTRACE;
    arealight.mesh = &arealightmesh;
    arealight.vertices[0] = float3(-1, -0.5, -1);
    arealight.vertices[1] = float3(-1, -0.5, 1);
    arealight.vertices[2] = float3(1, -0.5, 1);
    arealight.vertices[3] = float3(1, -0.5, -1);
    arealight.intensity = float3(0.1f, 0.9f, 0.5f);
    arealight.mesh->createSingleQuad(arealight.vertices);
    globalScene.addLight(&arealight);
    globalParticleConstraint = false;
    testCollisions = true;
    globalEnableParticles = true;
    globalParticleSystem.sphereMeshFilePath = "../../media/smallsphere.obj";
    globalParticleSystem.initialize();
}

static void setup3(int argc, const char* argv[]) {
    // bvh
    setupScene(argc, argv);
    globalRenderType = RENDER_RAYTRACE;
    light.position = float3(0.5f, 4.0f, 1.0f);
    light.wattage = float3(1000.0f, 1000.0f, 1000.0f);
    globalScene.addLight(&light);
}

static void setup4(int argc, const char* argv[]) {
    // area light/fresnel
    setupScene(argc, argv);
    globalRenderType = RENDER_RAYTRACE;
    arealight.mesh = &arealightmesh;
    arealight.vertices[0] = float3(-0.1, 0.68, -0.1);
    arealight.vertices[1] = float3(-0.1, 0.68, 0.1);
    arealight.vertices[2] = float3(0.1, 0.68, 0.1);
    arealight.vertices[3] = float3(0.1, 0.68, -0.1);
    arealight.intensity = float3(100.1f, 100.9f, 100.5f);
    arealight.mesh->createSingleQuad(arealight.vertices);
    globalScene.addLight(&arealight);
    arealight1.mesh = &arealightmesh1;
    arealight1.vertices[0] = float3(6.0f, 1.0f, 0.0f);
    arealight1.vertices[1] = float3(6.0f, 0.0f, 2.0f);
    arealight1.vertices[2] = float3(6.0f, 3.0f, 2.0f);
    arealight1.vertices[3] = float3(6.0f, 2.0f, 0.0f);
    arealight1.intensity = float3(50.1f, 100.9f, 100.5f);
    arealight1.mesh->createSingleQuad(arealight1.vertices);
    globalScene.addLight(&arealight1);
}


static void project(int argc, const char* argv[]) {
    // final product
    // run with custom cornellbox-glass.obj file
    //generateSpectrumXYZLookup(); // for spectral (incomplete)
    setupScene(argc, argv);
    setupLightSource();
    globalRenderType = RENDER_RAYTRACE;
    globalEnableParticles = true;
    globalParticleConstraint = true;
    testCollisions = false;
    globalParticleSystem.sphereMeshFilePath = "../../media/smallsphere.obj";
    globalParticleSystem.initialize();
}


int main(int argc, const char* argv[]) {
    srand(time(NULL));
    //setup1(argc, argv);
    //setup2();
    //setup3(argc, argv);
    //setup4(argc, argv);
    project(argc, argv);

    CS488.start();
}
