# eerdemYlmzz.github.io

Erdem YILMAZ'ın blogu. Jekyll ile kurulu, GitHub Pages üzerinde yayınlanıyor.

## Yeni yazı eklemek

`_posts/` klasörüne `YYYY-MM-DD-yazi-adi.md` formatında bir dosya aç ve başına şu front matter'ı koy:

```yaml
---
layout: post
published: true
title: Yazının Başlığı
date: 2026-08-15
categories: [kategori1, kategori2]
---
```

Dosyanın geri kalanı Markdown. Görseller `assets/images/` altına konur, yazının içinden `/assets/images/dosya.png` şeklinde çağrılır.

Her yazı kendi sayfasında yayınlanır (`/yazi-adi/`), ana sayfada sadece başlığı listelenir.

## Lokal çalıştırmak

```bash
bundle exec jekyll serve
```
