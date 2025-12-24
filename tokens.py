class Tokenizer(object):
    """
    一个用于 SMILES 字符串的简单分词器（Tokenizer）。

    支持：
    - 多字符 token（如 'Cl', 'Br', '@@' 等）
    - 处理方括号内原子（[C@@H] 等）
    - 特殊工具 token：<start>, <end>, <pad>

    内部维护两个字典：
    - tokensDict: token -> id
    - tokensInvDict: id -> token
    """
    def __init__(self, multiCharTokens, handleBraces=True,
                 toolTokens=('<start>', '<end>', '<pad>')):
        """
        参数
        ----
        multiCharTokens : list[str]
            SMILES 中的多字符 token 列表，例如 ['Cl','Br','@@'] 等。
        handleBraces : bool
            是否特殊处理方括号内的内容（例如 [C@@H] 作为一个整体 token）。
        toolTokens : tuple
            特殊工具 token，一般用于序列边界和 padding。
            默认顺序：('<start>', '<end>', '<pad>')
            注意：后续 getSmiles 假设它们的 id 分别为 0,1,2。
        """
        self.tokensDict, self.tokensInvDict = dict(), dict()
        self.toolTokens = toolTokens
        self.handleBraces = handleBraces

        # multiCharTokens 统一转小写，并按长度从长到短排序
        # 这样在匹配时可以优先匹配更长的 token，避免 'Cl' 被拆成 'C','l'。
        self.multiCharTokens = [
            t.lower()
            for t in sorted(multiCharTokens, key=lambda s: len(s), reverse=True)
        ]

        # 先把工具 token 加入词表，保证它们的 id 为 0,1,2,...
        for token in self.toolTokens:
            self.addToken(token)
    
    def addToken(self, token):
        """
        将一个新 token 加入词表（如果尚未出现）。
        """
        if token not in self.tokensDict:
            num = len(self.tokensDict)
            self.tokensDict[token] = num        # token -> id
            self.tokensInvDict[num] = token      # id -> token

    def tokenize(self, smilesStrs, useTokenDict=False):
        """
        将一批 SMILES 字符串拆分为 token 序列（字符/多字符）。

        参数
        ----
        smilesStrs : list[str]
            一批 SMILES 字符串。
        useTokenDict : bool
            如果为 True，则仅在已有的 tokensDict 中查找多字符 token；
            如果为 False，则使用初始化时给定的 multiCharTokens 列表。

        返回
        ----
        vectors : list[list[str]]
            每个 SMILES 被拆分成 token 列表，例如：
            'CC(=O)O' -> ['C','C','(', '=','O',')','O']
        """
        vectors = []

        if useTokenDict:
            # 在已有词表中，筛选出非工具 token 且长度>1 的多字符 token
            # 并按长度从长到短排序
            appliedMultiCharToken = sorted(
                [
                    key for key in self.tokensDict.keys()
                    if key not in self.toolTokens and len(key) > 1
                ],
                key=lambda s: len(s),
                reverse=True
            )
        else:
            # 使用初始化时传入的 multiCharTokens
            appliedMultiCharToken = self.multiCharTokens

        # 对每一个 SMILES 字符串进行分词
        for smilesStr in smilesStrs:
            currentVector = []
            startIdx, endIdx, length = 0, 0, len(smilesStr)

            while startIdx < length:
                foundMultiCharToken = False

                # 情况 1：如果开启 handleBraces，且当前字符是 '['，
                # 则找到匹配的 ']'，把整个 [ ... ] 作为一个 token。
                if self.handleBraces and not useTokenDict and smilesStr[startIdx] == '[':
                    endIdx = smilesStr.index(']', startIdx) + 1
                    foundMultiCharToken = True
                else:
                    # 情况 2：尝试匹配多字符 token
                    for token in appliedMultiCharToken:
                        tokenLen = len(token)
                        endIdx = startIdx + tokenLen
                        if endIdx > length:
                            # 超出边界，跳过
                            pass
                        else:
                            # useTokenDict=False 时，大小写不敏感（统一转小写匹配）
                            # useTokenDict=True 时，严格按原样匹配
                            if ((not useTokenDict and smilesStr[startIdx:endIdx].lower() == token)
                                    or (useTokenDict and smilesStr[startIdx:endIdx] == token)):
                                foundMultiCharToken = True
                                break

                # 如果没找到多字符 token，则当前字符单独作为一个 token
                if not foundMultiCharToken:
                    endIdx = startIdx + 1

                # 记录当前 token（子串）
                currentVector.append(smilesStr[startIdx:endIdx])
                # 移动指针
                startIdx = endIdx

            vectors.append(currentVector)

        return vectors
    
    def getTokensSize(self):
        """
        返回当前词表大小。
        """
        return len(self.tokensDict)
    
    def getTokensNum(self, token):
        """
        返回给定 token 对应的 id。
        """
        return self.tokensDict[token]

    def getNumVector(self, vectors, addStart=False, addEnd=False):
        """
        将 token 序列列表转换为 id 序列列表。

        参数
        ----
        vectors : list[list[str]]
            已分好词的 token 序列列表。
        addStart : bool
            是否在序列开头添加 <start>。
        addEnd : bool
            是否在序列末尾添加 <end>。

        返回
        ----
        numVector : list[list[int]]
            对应的 id 序列。
        """
        numVector = []
        for vec in vectors:
            currentVec = []
            # 开头添加 <start>
            if addStart:
                currentVec.append(self.tokensDict['<start>'])
            # 中间正常 token
            for elem in vec:
                currentVec.append(self.tokensDict[elem])
            # 末尾添加 <end>
            if addEnd:
                currentVec.append(self.tokensDict['<end>'])
            numVector.append(currentVec)

        return numVector

    def getSmiles(self, numVectors):
        """
        将一批 id 序列还原为 SMILES 字符串。
        用于从模型输出的 token id 序列中恢复 SMILES。

        假设：
        - <start> 的 id 为 0
        - <end>   的 id 为 1
        - <pad>   的 id 为 2

        解码逻辑：
        - 从后往前去掉所有尾部的 <pad>(id=2)
        - 若末尾为 <end>(id=1)，则去掉
        - 若开头为 <start>(id=0)，则去掉
        - 剩余 id 通过 tokensInvDict 还原为字符串并拼接
        """
        smileslist = []
        for numVector in numVectors:
            numVector = numVector.tolist()

            # 1) 去掉尾部的 <pad>（id=2）
            for i in range(len(numVector) - 1, -1, -1):
                if numVector[i] != 2:
                    break
                else:
                    numVector.pop(i)

            # 2) 若最后一个为 <end>（id=1），去掉
            if len(numVector) > 0 and numVector[-1] == 1:
                numVector.pop()

            # 3) 若第一个为 <start>（id=0），去掉
            if len(numVector) > 0 and numVector[0] == 0:
                numVector.pop(0)

            # 4) 将剩余的 token id 还原为实际 token 字符串并拼接
            smileslist.append(''.join([self.tokensInvDict[n] for n in numVector]))

        return smileslist
    
    def getInputNumVector(self, numVectors):
        """
        与 getSmiles 完全相同的逻辑（当前实现是重复代码），
        可能原意是用作“只解码输入序列”的接口。

        逻辑同样是：
        - 去尾部 <pad> (id=2)
        - 去尾部 <end> (id=1)
        - 去头部 <start> (id=0)
        - 还原 token 并拼接成字符串
        """
        smileslist = []
        for numVector in numVectors:
            numVector = numVector.tolist()
            for i in range(len(numVector) - 1, -1, -1):
                if numVector[i] != 2:
                    break
                else:
                    numVector.pop(i)
            if len(numVector) > 0 and numVector[-1] == 1:
                numVector.pop()
            if len(numVector) > 0 and numVector[0] == 0:
                numVector.pop(0)
            smileslist.append(''.join([self.tokensInvDict[n] for n in numVector]))
        return smileslist


def getTokenizer(file, handleBraces=True):
    """
    根据给定 SMILES 文件动态构建一个 Tokenizer：
    - 初始化时先加入常见多字符原子 token
    - 然后遍历文件中所有 SMILES，将出现过的 token 加入词表

    参数
    ----
    file : str
        存放 SMILES 的文本文件路径，每行一个 SMILES。
    handleBraces : bool
        是否在 tokenizer 中处理方括号原子。

    返回
    ----
    tokenizer : Tokenizer
        构建好的分词器，包含完整词表。
    """
    # 预定义的多字符 token 列表（常见元素符号 + '@@' 手性标记）
    tokenizer = Tokenizer(
        ['Li', 'Be', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ca',
         'As', 'Se', 'Br', 'Te', '@@'],
        handleBraces=handleBraces
    )

    with open(file, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:   # EOF
                break

            # 跳过 FASTA 样式的注释行，确保蛋白质序列数据不会把标签加入词表
            stripped = line.strip()
            if not stripped or stripped.startswith('>'):
                continue

            # 对当前行做分词，拿到该行的 token 列表
            # 注意：tokenize([stripped]) 返回的是 list[list[str]]，
            # 一般需要取第 0 个元素。
            tokens = set(*tokenizer.tokenize([stripped]))
            # 更稳妥的写法建议改为：
            # tokens = set(tokenizer.tokenize([stripped])[0])

            # 将该行中所有出现过的 token 加入词表
            for token in tokens:
                tokenizer.addToken(token)

    return tokenizer
