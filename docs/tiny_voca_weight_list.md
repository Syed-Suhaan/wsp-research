Checkpoint: tiny-voca.pt  (training step 104737, format version 4)
184 tensors, 4,537,291 parameters

| Name | Shape | Parameters |
|---|---|---:|
| `tokens.weight` | [76, 192] | 14,592 |
| `stress.weight` | [3, 192] | 576 |
| `voices.weight` | [2, 192] | 384 |
| `encoder.blocks.0.norm1.weight` | [192] | 192 |
| `encoder.blocks.0.norm1.bias` | [192] | 192 |
| `encoder.blocks.0.attention.qkv.weight` | [576, 192] | 110,592 |
| `encoder.blocks.0.attention.qkv.bias` | [576] | 576 |
| `encoder.blocks.0.attention.out.weight` | [192, 192] | 36,864 |
| `encoder.blocks.0.attention.out.bias` | [192] | 192 |
| `encoder.blocks.0.norm2.weight` | [192] | 192 |
| `encoder.blocks.0.norm2.bias` | [192] | 192 |
| `encoder.blocks.0.depthwise.weight` | [192, 1, 7] | 1,344 |
| `encoder.blocks.0.depthwise.bias` | [192] | 192 |
| `encoder.blocks.0.expand.weight` | [640, 192] | 122,880 |
| `encoder.blocks.0.expand.bias` | [640] | 640 |
| `encoder.blocks.0.project.weight` | [192, 640] | 122,880 |
| `encoder.blocks.0.project.bias` | [192] | 192 |
| `encoder.blocks.1.norm1.weight` | [192] | 192 |
| `encoder.blocks.1.norm1.bias` | [192] | 192 |
| `encoder.blocks.1.attention.qkv.weight` | [576, 192] | 110,592 |
| `encoder.blocks.1.attention.qkv.bias` | [576] | 576 |
| `encoder.blocks.1.attention.out.weight` | [192, 192] | 36,864 |
| `encoder.blocks.1.attention.out.bias` | [192] | 192 |
| `encoder.blocks.1.norm2.weight` | [192] | 192 |
| `encoder.blocks.1.norm2.bias` | [192] | 192 |
| `encoder.blocks.1.depthwise.weight` | [192, 1, 7] | 1,344 |
| `encoder.blocks.1.depthwise.bias` | [192] | 192 |
| `encoder.blocks.1.expand.weight` | [640, 192] | 122,880 |
| `encoder.blocks.1.expand.bias` | [640] | 640 |
| `encoder.blocks.1.project.weight` | [192, 640] | 122,880 |
| `encoder.blocks.1.project.bias` | [192] | 192 |
| `encoder.blocks.2.norm1.weight` | [192] | 192 |
| `encoder.blocks.2.norm1.bias` | [192] | 192 |
| `encoder.blocks.2.attention.qkv.weight` | [576, 192] | 110,592 |
| `encoder.blocks.2.attention.qkv.bias` | [576] | 576 |
| `encoder.blocks.2.attention.out.weight` | [192, 192] | 36,864 |
| `encoder.blocks.2.attention.out.bias` | [192] | 192 |
| `encoder.blocks.2.norm2.weight` | [192] | 192 |
| `encoder.blocks.2.norm2.bias` | [192] | 192 |
| `encoder.blocks.2.depthwise.weight` | [192, 1, 7] | 1,344 |
| `encoder.blocks.2.depthwise.bias` | [192] | 192 |
| `encoder.blocks.2.expand.weight` | [640, 192] | 122,880 |
| `encoder.blocks.2.expand.bias` | [640] | 640 |
| `encoder.blocks.2.project.weight` | [192, 640] | 122,880 |
| `encoder.blocks.2.project.bias` | [192] | 192 |
| `encoder.blocks.3.norm1.weight` | [192] | 192 |
| `encoder.blocks.3.norm1.bias` | [192] | 192 |
| `encoder.blocks.3.attention.qkv.weight` | [576, 192] | 110,592 |
| `encoder.blocks.3.attention.qkv.bias` | [576] | 576 |
| `encoder.blocks.3.attention.out.weight` | [192, 192] | 36,864 |
| `encoder.blocks.3.attention.out.bias` | [192] | 192 |
| `encoder.blocks.3.norm2.weight` | [192] | 192 |
| `encoder.blocks.3.norm2.bias` | [192] | 192 |
| `encoder.blocks.3.depthwise.weight` | [192, 1, 7] | 1,344 |
| `encoder.blocks.3.depthwise.bias` | [192] | 192 |
| `encoder.blocks.3.expand.weight` | [640, 192] | 122,880 |
| `encoder.blocks.3.expand.bias` | [640] | 640 |
| `encoder.blocks.3.project.weight` | [192, 640] | 122,880 |
| `encoder.blocks.3.project.bias` | [192] | 192 |
| `encoder.norm.weight` | [192] | 192 |
| `encoder.norm.bias` | [192] | 192 |
| `duration.conv1.weight` | [128, 192, 3] | 73,728 |
| `duration.conv1.bias` | [128] | 128 |
| `duration.conv2.weight` | [128, 128, 3] | 49,152 |
| `duration.conv2.bias` | [128] | 128 |
| `duration.norm1.weight` | [128] | 128 |
| `duration.norm1.bias` | [128] | 128 |
| `duration.norm2.weight` | [128] | 128 |
| `duration.norm2.bias` | [128] | 128 |
| `duration.out.weight` | [2, 128] | 256 |
| `duration.out.bias` | [2] | 2 |
| `frame_voices.weight` | [2, 192] | 384 |
| `frame_position.weight` | [192, 3] | 576 |
| `frame_position.bias` | [192] | 192 |
| `prosody_stack.blocks.0.norm1.weight` | [192] | 192 |
| `prosody_stack.blocks.0.norm1.bias` | [192] | 192 |
| `prosody_stack.blocks.0.attention.qkv.weight` | [576, 192] | 110,592 |
| `prosody_stack.blocks.0.attention.qkv.bias` | [576] | 576 |
| `prosody_stack.blocks.0.attention.out.weight` | [192, 192] | 36,864 |
| `prosody_stack.blocks.0.attention.out.bias` | [192] | 192 |
| `prosody_stack.blocks.0.norm2.weight` | [192] | 192 |
| `prosody_stack.blocks.0.norm2.bias` | [192] | 192 |
| `prosody_stack.blocks.0.depthwise.weight` | [192, 1, 7] | 1,344 |
| `prosody_stack.blocks.0.depthwise.bias` | [192] | 192 |
| `prosody_stack.blocks.0.expand.weight` | [640, 192] | 122,880 |
| `prosody_stack.blocks.0.expand.bias` | [640] | 640 |
| `prosody_stack.blocks.0.project.weight` | [192, 640] | 122,880 |
| `prosody_stack.blocks.0.project.bias` | [192] | 192 |
| `prosody_stack.blocks.1.norm1.weight` | [192] | 192 |
| `prosody_stack.blocks.1.norm1.bias` | [192] | 192 |
| `prosody_stack.blocks.1.attention.qkv.weight` | [576, 192] | 110,592 |
| `prosody_stack.blocks.1.attention.qkv.bias` | [576] | 576 |
| `prosody_stack.blocks.1.attention.out.weight` | [192, 192] | 36,864 |
| `prosody_stack.blocks.1.attention.out.bias` | [192] | 192 |
| `prosody_stack.blocks.1.norm2.weight` | [192] | 192 |
| `prosody_stack.blocks.1.norm2.bias` | [192] | 192 |
| `prosody_stack.blocks.1.depthwise.weight` | [192, 1, 7] | 1,344 |
| `prosody_stack.blocks.1.depthwise.bias` | [192] | 192 |
| `prosody_stack.blocks.1.expand.weight` | [640, 192] | 122,880 |
| `prosody_stack.blocks.1.expand.bias` | [640] | 640 |
| `prosody_stack.blocks.1.project.weight` | [192, 640] | 122,880 |
| `prosody_stack.blocks.1.project.bias` | [192] | 192 |
| `prosody_stack.norm.weight` | [192] | 192 |
| `prosody_stack.norm.bias` | [192] | 192 |
| `prosody_head.weight` | [3, 192] | 576 |
| `prosody_head.bias` | [3] | 3 |
| `prosody_projection.weight` | [192, 3] | 576 |
| `prosody_projection.bias` | [192] | 192 |
| `acoustic_stack.blocks.0.norm1.weight` | [192] | 192 |
| `acoustic_stack.blocks.0.norm1.bias` | [192] | 192 |
| `acoustic_stack.blocks.0.attention.qkv.weight` | [576, 192] | 110,592 |
| `acoustic_stack.blocks.0.attention.qkv.bias` | [576] | 576 |
| `acoustic_stack.blocks.0.attention.out.weight` | [192, 192] | 36,864 |
| `acoustic_stack.blocks.0.attention.out.bias` | [192] | 192 |
| `acoustic_stack.blocks.0.norm2.weight` | [192] | 192 |
| `acoustic_stack.blocks.0.norm2.bias` | [192] | 192 |
| `acoustic_stack.blocks.0.depthwise.weight` | [192, 1, 7] | 1,344 |
| `acoustic_stack.blocks.0.depthwise.bias` | [192] | 192 |
| `acoustic_stack.blocks.0.expand.weight` | [640, 192] | 122,880 |
| `acoustic_stack.blocks.0.expand.bias` | [640] | 640 |
| `acoustic_stack.blocks.0.project.weight` | [192, 640] | 122,880 |
| `acoustic_stack.blocks.0.project.bias` | [192] | 192 |
| `acoustic_stack.blocks.1.norm1.weight` | [192] | 192 |
| `acoustic_stack.blocks.1.norm1.bias` | [192] | 192 |
| `acoustic_stack.blocks.1.attention.qkv.weight` | [576, 192] | 110,592 |
| `acoustic_stack.blocks.1.attention.qkv.bias` | [576] | 576 |
| `acoustic_stack.blocks.1.attention.out.weight` | [192, 192] | 36,864 |
| `acoustic_stack.blocks.1.attention.out.bias` | [192] | 192 |
| `acoustic_stack.blocks.1.norm2.weight` | [192] | 192 |
| `acoustic_stack.blocks.1.norm2.bias` | [192] | 192 |
| `acoustic_stack.blocks.1.depthwise.weight` | [192, 1, 7] | 1,344 |
| `acoustic_stack.blocks.1.depthwise.bias` | [192] | 192 |
| `acoustic_stack.blocks.1.expand.weight` | [640, 192] | 122,880 |
| `acoustic_stack.blocks.1.expand.bias` | [640] | 640 |
| `acoustic_stack.blocks.1.project.weight` | [192, 640] | 122,880 |
| `acoustic_stack.blocks.1.project.bias` | [192] | 192 |
| `acoustic_stack.blocks.2.norm1.weight` | [192] | 192 |
| `acoustic_stack.blocks.2.norm1.bias` | [192] | 192 |
| `acoustic_stack.blocks.2.attention.qkv.weight` | [576, 192] | 110,592 |
| `acoustic_stack.blocks.2.attention.qkv.bias` | [576] | 576 |
| `acoustic_stack.blocks.2.attention.out.weight` | [192, 192] | 36,864 |
| `acoustic_stack.blocks.2.attention.out.bias` | [192] | 192 |
| `acoustic_stack.blocks.2.norm2.weight` | [192] | 192 |
| `acoustic_stack.blocks.2.norm2.bias` | [192] | 192 |
| `acoustic_stack.blocks.2.depthwise.weight` | [192, 1, 7] | 1,344 |
| `acoustic_stack.blocks.2.depthwise.bias` | [192] | 192 |
| `acoustic_stack.blocks.2.expand.weight` | [640, 192] | 122,880 |
| `acoustic_stack.blocks.2.expand.bias` | [640] | 640 |
| `acoustic_stack.blocks.2.project.weight` | [192, 640] | 122,880 |
| `acoustic_stack.blocks.2.project.bias` | [192] | 192 |
| `acoustic_stack.blocks.3.norm1.weight` | [192] | 192 |
| `acoustic_stack.blocks.3.norm1.bias` | [192] | 192 |
| `acoustic_stack.blocks.3.attention.qkv.weight` | [576, 192] | 110,592 |
| `acoustic_stack.blocks.3.attention.qkv.bias` | [576] | 576 |
| `acoustic_stack.blocks.3.attention.out.weight` | [192, 192] | 36,864 |
| `acoustic_stack.blocks.3.attention.out.bias` | [192] | 192 |
| `acoustic_stack.blocks.3.norm2.weight` | [192] | 192 |
| `acoustic_stack.blocks.3.norm2.bias` | [192] | 192 |
| `acoustic_stack.blocks.3.depthwise.weight` | [192, 1, 7] | 1,344 |
| `acoustic_stack.blocks.3.depthwise.bias` | [192] | 192 |
| `acoustic_stack.blocks.3.expand.weight` | [640, 192] | 122,880 |
| `acoustic_stack.blocks.3.expand.bias` | [640] | 640 |
| `acoustic_stack.blocks.3.project.weight` | [192, 640] | 122,880 |
| `acoustic_stack.blocks.3.project.bias` | [192] | 192 |
| `acoustic_stack.blocks.4.norm1.weight` | [192] | 192 |
| `acoustic_stack.blocks.4.norm1.bias` | [192] | 192 |
| `acoustic_stack.blocks.4.attention.qkv.weight` | [576, 192] | 110,592 |
| `acoustic_stack.blocks.4.attention.qkv.bias` | [576] | 576 |
| `acoustic_stack.blocks.4.attention.out.weight` | [192, 192] | 36,864 |
| `acoustic_stack.blocks.4.attention.out.bias` | [192] | 192 |
| `acoustic_stack.blocks.4.norm2.weight` | [192] | 192 |
| `acoustic_stack.blocks.4.norm2.bias` | [192] | 192 |
| `acoustic_stack.blocks.4.depthwise.weight` | [192, 1, 7] | 1,344 |
| `acoustic_stack.blocks.4.depthwise.bias` | [192] | 192 |
| `acoustic_stack.blocks.4.expand.weight` | [640, 192] | 122,880 |
| `acoustic_stack.blocks.4.expand.bias` | [640] | 640 |
| `acoustic_stack.blocks.4.project.weight` | [192, 640] | 122,880 |
| `acoustic_stack.blocks.4.project.bias` | [192] | 192 |
| `acoustic_stack.norm.weight` | [192] | 192 |
| `acoustic_stack.norm.bias` | [192] | 192 |
| `vocoder.harmonic.weight` | [98, 192] | 18,816 |
| `vocoder.harmonic.bias` | [98] | 98 |
| `vocoder.noise.weight` | [36, 192] | 6,912 |
| `vocoder.noise.bias` | [36] | 36 |
