## 1. 接口 app-save-speech ##
1. 保存speech的 tense   -----  文件 .pt
2. 文件命令 区分 ------  通过 userId + '-' +  创建时间的 getTime()


## 2. 接口 app-output-speech ##
1. 读取pt 文件 ------  通过 userId + '-' +  创建时间的 getTime()
2. 合成部分音频， 生成在本地  -->  app循环读取文件，可播放  -->   直到合成完毕，读取最后的文件  -->   下载到APP 本地
    (1). 