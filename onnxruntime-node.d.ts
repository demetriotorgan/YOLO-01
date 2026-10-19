declare module 'onnxruntime-node' {
    export class Tensor {
        constructor(type: string, data: Float32Array | number[] | Float64Array | Int32Array, dims: number[]);
        data: any;
        dims: readonly number[];
    }

    export class InferenceSession {
        inputNames: string[];
        outputNames: string[];
        run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
        static create(modelPath: string): Promise<InferenceSession>;
    }
}