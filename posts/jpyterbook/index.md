---
title: "如何使用 Python jupyter-book 建立一本書"
author: "紙魚"
date: "2025-07-25"
categories: [other, python]
image: "logo.png"
---

# 前言

說到 jupyter-book，可能很多人會想到 Jupyter Notebook，但這裡指的是一個 python 套件，用來建立線上書籍的工具，特別適合用於副檔名為 `.ipynb` 的 Jupyter Notebook 。最後產出的網站類似 gitbook 或是 bookdown 的靜態網站，適合用來寫教學、筆記或是書籍。也是在我使用 Quarto 之前考慮過的選項之一，雖然最後因為產出有點陽春、能改動的東西較少所以放棄了，但它的製作過程簡單，所以留下這個筆記，方便日後需要時可以快速上手。

## 使用環境

這是我的使用環境，不一定適用每個人，但截至目前為止都運作順利：

- IDE ： VsCode 

- Python version : 3.12.5

# 首次興建

依照以下步驟操作：

---

## 安裝 jupyter-book

```bash
pip install -U jupyter-book
```

---

## 建立一本書的骨架

使用命令行建立一個新的書籍專案：

```bash
jupyter-book create mybook/
```

這會建立一個名為 `mybook/` 的資料夾，裡面包含書籍的基本結構，包括 Markdown 與 Jupyter Notebook 範例，`mybook/` 亦可以自行換成其他資料夾名稱。

---

##  編輯內容

> 從這裡開始就是在編輯時會重複執行的步驟！


可以在 `mybook/` 目錄中看到這些重要檔案和資料夾：

* `mybook/_config.yml`：書籍的設定（標題、主題、logo 等）
* `mybook/_toc.yml`：書籍目錄（控制章節順序）
* `mybook/intro.md`、`mybook/chapters/*.ipynb`：實際內容，可新增 Markdown 或 Notebook 檔案

### 新增章節例子：

1. 在 `mybook/` 下新增一個檔案，例如 `chapter1.md`
2. 在 `_toc.yml` 中加上該檔案的設定：

```yaml
format: jb-book
root: intro
chapters:
  - file: chapter1
```

---

## 編譯網站


在書籍資料夾中執行：

```bash
jupyter-book build mybook/
```

這會自動產生靜態網站，輸出目錄為：

```
mybook/_build/html/
```

---

##  預覽網站

使用瀏覽器打開以下檔案即可：

```
mybook/_build/html/index.html
```

或用 Python 的 HTTP server 預覽：

```bash
cd mybook/_build/html
python -m http.server
```

然後打開瀏覽器到 [http://localhost:8000](http://localhost:8000)

> 1. 注意：如果你有修改 `_toc.yml` 或其他設定，必須重新編譯網站才能看到變更，不然會顯示舊的內容。
> 2. 如果編譯網站後有做**刪除頁面**的動作，請到 `_build/html/` 資料夾刪除對應的 html，不然他不會蓋到~~很笨我知道~~。
> 3. 執行後 terminal 的 powershell 非必要請不要關閉，因為它在運行 HTTP server。需要關閉可以在 terminal 視窗 輸入 `Ctrl + C` 停止服務。或是另開新的 terminal 視窗來執行其他命令。

---

##  發布網站（選擇性）

可以把 `_build/html` 上傳到 GitHub Pages、Netlify、Vercel 等平台，或用 GitHub Actions 自動部署。


