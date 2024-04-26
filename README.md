# Action Policy
This is a repo for training the action policy

# Log
## 2024/04/26
1. Calvin评测环境需要OpenGL, 安装是安装了，但是出现了bug, 参考https://github.com/RoboFlamingo/RoboFlamingo/issues/2, 不知道怎么解决
File "/home/yupeiqi/miniconda3/envs/RoboFlamingo/lib/python3.8/site-packages/pyrender/shader_program.py", line 137, in _add_to_context
    self._program_id = gl_shader_utils.compileProgram(*shader_ids)
  File "/home/yupeiqi/pyopengl/OpenGL/GL/shaders.py", line 206, in compileProgram
    program.check_validate()
  File "/home/yupeiqi/pyopengl/OpenGL/GL/shaders.py", line 106, in check_validate
    raise RuntimeError(
RuntimeError: Validation failure (0):
2. 转战Droid数据集！ 