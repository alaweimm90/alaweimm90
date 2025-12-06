# Attributa.dev Deployment Guide

## Quick Deploy with Lovable

The fastest way to deploy Attributa.dev is through Lovable's built-in deployment:

1. **One-Click Deploy**: Click the "Publish" button in Lovable
2. **Custom Domain**: Add your domain in Project Settings → Domains
3. **SSL**: Automatic HTTPS with custom domains

## Manual Deployment Options

### Static Site Hosting

Since Attributa.dev is a fully client-side application with local processing:

**Vercel**
```bash
npm run build
vercel --prod
```

**Netlify**
```bash
npm run build
netlify deploy --prod --dir=dist
```

**GitHub Pages**
```bash
npm run build
# Push dist/ to gh-pages branch
```

### Environment Configuration

Create `.env.production`:
```env
# Optional: Custom Crossref User-Agent
VITE_CROSSREF_USER_AGENT=YourApp/1.0 (mailto:your@email.com)

# Optional: Analytics
VITE_ANALYTICS_ID=your-analytics-id
```

### Build Requirements

- **Node.js**: 18+ recommended
- **Memory**: 2GB+ for build process
- **Storage**: ~500MB for dependencies

### Performance Optimization

**Pre-compression**
```bash
# Enable gzip/brotli compression
npm install -g gzipper
gzipper compress ./dist --brotli
```

**CDN Configuration**
- Cache static assets for 1 year
- Cache HTML for 1 hour
- Enable compression (gzip/brotli)

### Security Headers

Add these headers for production:

```
Content-Security-Policy: default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline' fonts.googleapis.com; font-src 'self' fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self' api.crossref.org
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(), microphone=(), geolocation=()
```

### Database (Optional)

Attributa.dev works fully client-side, but you can add Supabase for:
- User authentication
- Settings synchronization
- Usage analytics

See `src/integrations/supabase/` for implementation details.

### Monitoring

**Performance Monitoring**
- Core Web Vitals tracking
- Bundle size analysis
- Model loading times

**Error Tracking**
- Client-side error reporting
- Model loading failures
- API rate limit issues

### Scaling Considerations

**High Traffic**
- Use CDN for static assets
- Implement service worker caching
- Consider edge computing for API endpoints

**Large Organizations**
- Deploy behind corporate firewall
- Configure internal model hosting
- Implement SSO integration

## Production Checklist

- [ ] Favicon and branding assets
- [ ] Custom domain with SSL
- [ ] Security headers configured
- [ ] Analytics implemented
- [ ] Error monitoring setup
- [ ] Performance optimization
- [ ] Backup strategy for user data
- [ ] Privacy policy and terms published
- [ ] Documentation updated

## Support

For deployment assistance:
- **Email**: support@attributa.dev
- **Documentation**: https://docs.attributa.dev
- **GitHub Issues**: https://github.com/alawein/Attributa

---

*Remember: Attributa.dev processes content locally, making it suitable for sensitive environments and privacy-conscious organizations.*