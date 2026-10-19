import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import fs from 'fs';

async function runYoloPhaseZero(): Promise<void> {
    try {
        console.log("🚀 [Fase 0] Iniciando motor YOLO local (Node.js + TypeScript)...");
        
        // 1. Verificar se o modelo ONNX existe
        const modelPath = './model/yolov8n.onnx';
        if (!fs.existsSync(modelPath)) {
            console.error(`❌ Erro: O arquivo do modelo não foi encontrado em ${modelPath}`);
            console.log("💡 Dica: Coloque o seu modelo YOLOv8 exportado em .onnx na pasta /model.");
            return;
        }

        const session = await ort.InferenceSession.create(modelPath);
        console.log("✅ Modelo YOLO carregado com sucesso!");

        // 2. Verificar se a imagem de teste existe
        const imagePath = './dataset/imagem1.jpeg';
        if (!fs.existsSync(imagePath)) {
            console.error(`❌ Erro: Nenhuma imagem de teste encontrada em ${imagePath}`);
            return;
        }

        const inputWidth = 640;
        const inputHeight = 640;

        console.log(`🖼️ Processando imagem da gôndola: ${imagePath}`);
        const imageBuffer = await sharp(imagePath)
            .resize(inputWidth, inputHeight, { fit: 'fill' })
            .raw()
            .toBuffer();

        // 3. Converter pixels para Tensor
        const tensorData = new Float32Array(3 * inputWidth * inputHeight);
        for (let h = 0; h < inputHeight; h++) {
            for (let w = 0; w < inputWidth; w++) {
                const srcIdx = (h * inputWidth + w) * 3;
                tensorData[0 * inputWidth * inputHeight + h * inputWidth + w] = imageBuffer[srcIdx] / 255.0;     // R
                tensorData[1 * inputWidth * inputHeight + h * inputWidth + w] = imageBuffer[srcIdx + 1] / 255.0; // G
                tensorData[2 * inputWidth * inputHeight + h * inputWidth + w] = imageBuffer[srcIdx + 2] / 255.0; // B
            }
        }

        const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, inputHeight, inputWidth]);

        // 4. Executar inferência local com segurança estrita de tipos
        console.log("🔍 Executando inferência de detecção de ROI...");
        
        const inputNames = session.inputNames;
        const outputNames = session.outputNames;

        if (!inputNames || inputNames.length === 0 || !outputNames || outputNames.length === 0) {
            console.error("❌ Erro: Nomes de entrada ou saída do modelo não encontrados.");
            return;
        }

        const inputName = inputNames[0] as string;
        const outputName = outputNames[0] as string;

        const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };
        const results = await session.run(feeds);
        
        const output = results[outputName];
        if (!output) {
            console.error("❌ Erro: O modelo não retornou resultados.");
            return;
        }

        console.log("🎯 Inferência concluída com sucesso!");
        console.log(`📦 Dimensões da matriz de saída (Bounding Boxes):`, output.dims);

    } catch (error) {
        console.error("❌ Erro durante a execução do YOLO na Fase 0:", error);
    }
}

runYoloPhaseZero();