# 卒業論文・修士論文 TeXテンプレート

## 環境

- Docker
    ```bash
    docker pull texlive/texlive:latest
    ```

- VSCode
  - [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)
  - [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)

## 使い方

1. このリポジトリをフォークしてからcloneする．
1. clone先の**ディレクトリ**を開く
1. [thesis.tex](./thesis.tex)に本文を書いていく．
1. **Build LaTeX project**内の**Recipe: latexmk(latexmkrc)**でビルドする．以降，保存するたびに自動でビルドされる．失敗時は，**Build LaTeX project**内の**Terminate current compilation**で中断する．

## サンプル

[thesis.pdf](./thesis.pdf)
