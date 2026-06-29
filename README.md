## Anki Flashcards Maker
<img width="15%" height="15%" alt="flashcards-maker" src="https://github.com/user-attachments/assets/d4e051f7-8f0b-467c-9893-479ee4cd6bd4" />

批量根据提示词生成 Anki 闪卡的小工具。需要配合 Obsidian 的 Yanki 插件使用。

Yanki 插件能够将一篇 Obsidan 笔记转换成 Anki 能够识读的闪卡 (Flashcard)，并且以文件夹为牌组。其基本语法是：在一篇笔记内，以 `---` 的分隔线为界，上半部分是闪卡的正面（问题面），下半部分为背面（答案面）。

本工具能够通过模型提供商调用模型 API，以便批量且多并发地生成闪卡。工具支持 Agent Platform (原 Vertex AI) 的 JSON 鉴权模式。

要想制作闪卡，需要在工具内填写适当的提示词。工具的初始用途是服务于语言学习。但根据输入框的性质，同样可以用于生成其他类型的知识。因为工具会将用户输入的信息，按照每行一条的顺序提取，并发送给模型制作词卡，因此知识类型不局限于单词。

<img width="1740" height="1132" alt="image" src="https://github.com/user-attachments/assets/73bdb245-0a10-4e02-8611-8d9668f3bb73" />

本工具本质上是一个聊天输入框。其实拿它干什么都可以。



