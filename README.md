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
  找了很多办法了，也解决不了这个问题。感觉是系统问题。
2. 转战Droid数据集！ 
  `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
  库的版本不兼容，decision_transformer源代码需要python3.8(transformers4.5.1只有3.8python才能安装上)
  而droid数据的读取代码需要python3.10（dlimp需要高版本tensorflow，低版本python安装不了）