from native import locate,answer_span
def prefixes(chunks):return ['']+[''.join(chunks[:i]) for i in range(1,len(chunks)+1)]
for chunks,mode in [(['{"总部城市":',' "','武汉','"}'],'exact'),(['{"总部城市','":"','武汉','"}'],'preceding_token')]:
 raw=''.join(chunks);b,e=locate('kimi',raw,'总部城市',prefixes(chunks));assert e is None and b['mode']==mode and len(b['prefix'])<=b['value_char']
 assert raw[b['value_char']]=='"'
raw=' to=self<|message|>{"总部城市":"错误示例"}<|eom|><|start|>assistant to=user<|message|>{"总部城市":null}'
b,e=locate('muse',raw,'总部城市',prefixes(list(raw)));assert e is None and b['prefix'].endswith('to=user<|message|>{"总部城市":')
raw='<think>{"总部城市":"错误示例"}</think><answer>{"总部城市":null}</answer>'
b,e=locate('glm',raw,'总部城市',prefixes(list(raw)));assert e is None and b['prefix'].endswith('<answer>{"总部城市":')
assert answer_span('glm','<think>还在思考') is None
raw='{"总部城市":null}<|im_end|>';span=answer_span('kimi',raw);assert raw[slice(*span)]=='{"总部城市":null}'
assert locate('kimi','null','总部城市',prefixes(['null']))[1]=='missing_target_key'
print('Native final-channel isolation and preceding-token boundary checks passed')
