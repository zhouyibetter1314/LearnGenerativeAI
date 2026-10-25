# LaTeX 书稿框架

主文件为 `book.tex`，使用 `ctexbook` 文档类与 XeLaTeX，中文字体采用 TeX Live 自带的 Fandol。当前只建立标题与目录层级，正文、公式、图表和自检题暂不迁移。

## 文件组织

- 前置部分：`00_reading_guide.tex`（阅读指南）、`00_notation.tex`（符号约定）。
- 正文：沿用 `01_intro.tex` 到 `08_diffusion.tex` 的八章顺序；`02_math.tex` 对应 Markdown 的概率建模章。
- 附录：`appendix_formulas.tex`（核心公式速查）、`appendix_sources.tex`（课件来源与勘误）。
- 每个内容文件的开头标明原 Markdown 来源，具体文件均位于 `content/`。

章节采用 `\chapter → \section → \subsection`；正文及附录的编号和目录显示到子节（例如 4.1.1）。阅读指南与符号约定作为不编号的前置部分。原 Markdown 的已有标题保持层级，正文中的独立主题、表格条目和折叠推导进一步拆为子节。各章“本章要点”不编号、不进入目录。

现有材料未展开的方法保留在“后续阅读”之下，暂不扩写为新的方法章节。每处 `% TODO` 都是不会显示在成书中的正文占位注释；其后的 `\contentplaceholder` 仅输出空白段落，确保空章节可以正常分页。填写正文时，用实际内容替换该占位命令。

## 编译

先进入本目录，再执行：

```sh
cd book
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build book.tex
```

输出为 `build/book.pdf`。也可以连续运行两次 XeLaTeX：

```sh
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build book.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build book.tex
```

需要 TeX Live 中的中文支持（ctex、Fandol）及主文件所用的常规宏包。编辑器中将 `book.tex` 设为根文档，并将编译工作目录设为 `book/`。

## 后续填写

- 保留已有语义标签，使用 `\ref{sec:mle-kl}` 等进行交叉引用，编号由 LaTeX 自动生成。
- 图片放在仓库的 `figures/` 中，主文件已配置该路径。
- 如需继续细分，可使用 `\subsubsection`；默认不编号且不出现在目录中。
- 本章参考阅读预留在各章末尾，完整资料映射与勘误集中在附录。
