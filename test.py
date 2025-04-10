# import pathlib
# import openwakeword
# import openwakeword.utils


# F = openwakeword.utils.AudioFeatures()

# print(F)

import onnx

model_path = "/home/openWakeWord/openwakeword/resources/models/embedding_model.onnx"
model = onnx.load(model_path)

# # 모델의 연산 노드 전체 출력
# for i, node in enumerate(model.graph.node):
#     print(f"{i}: {node.op_type} - {node.name}")

# Shape inference 수행
inferred_model = onnx.shape_inference.infer_shapes(model)

# 출력 텐서 정보 확인
print("🔹 Output tensors after shape inference:")
for output in inferred_model.graph.output:
    print(output.name)
    for dim in output.type.tensor_type.shape.dim:
        print(f"  Dim: {dim.dim_value}")