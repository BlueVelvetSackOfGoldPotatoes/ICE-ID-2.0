# GitHub Pages Setup for ICE-ID

## Quick Setup (5 minutes)

### Enable GitHub Pages

1. Go to your repository: https://github.com/BlueVelvetSackOfGoldPotatoes/iceid
2. Click **Settings** → **Pages** (in sidebar)
3. Under "Build and deployment":
   - Source: **"Deploy from a branch"**
   - Branch: **"main"**
   - Folder: **"/docs"**
4. Click **Save**

### Your Site URL

After 1-2 minutes, your site will be live at:
```
https://bluevelvetsackofgoldpotatoes.github.io/iceid/
```

---

## What's Included

The `index.html` file is a complete, self-contained website with:

- ✅ Modern responsive design
- ✅ Gradient header with CTAs
- ✅ Feature cards and model badges
- ✅ Quick start guide
- ✅ Smooth scroll navigation
- ✅ SEO-friendly meta tags
- ✅ No build step required!

---

## Updating the Website

1. Edit `docs/index.html`
2. Commit and push:
   ```bash
   git add docs/index.html
   git commit -m "Update website"
   git push origin main
   ```
3. GitHub auto-rebuilds (1-2 min)

---

## Optional: Custom Domain

Create `docs/CNAME` with your domain:
```
iceid.yourdomain.com
```

Then add a CNAME DNS record pointing to `bluevelvetsackofgoldpotatoes.github.io`

---

## Troubleshooting

- **Site not showing?** Check Settings → Pages and wait 2 minutes
- **404 errors?** Verify `/docs` folder is selected
- **Changes not appearing?** Force refresh: Ctrl+Shift+R

---

For more help, see [GitHub Pages Docs](https://docs.github.com/en/pages)

