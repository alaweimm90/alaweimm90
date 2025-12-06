import React from 'react';

export const PWAManifest = () => {
  const manifest = {
    "name": "QMLab - Quantum Machine Learning Playground",
    "short_name": "QMLab",
    "description": "Open-source quantum machine learning playground for education and research",
    "start_url": "/",
    "display": "standalone",
    "theme_color": "#00BFFF",
    "background_color": "#0A0F1C",
    "orientation": "portrait-primary",
    "categories": ["education", "science", "productivity"],
    "icons": [
      {
        "src": "/icons/qmlab-192.png",
        "sizes": "192x192",
        "type": "image/png",
        "purpose": "any"
      },
      {
        "src": "/icons/qmlab-512.png", 
        "sizes": "512x512",
        "type": "image/png",
        "purpose": "any"
      },
      {
        "src": "/icons/qmlab-512-maskable.png",
        "sizes": "512x512", 
        "type": "image/png",
        "purpose": "maskable"
      }
    ],
    "shortcuts": [
      {
        "name": "Launch Playground",
        "short_name": "Playground",
        "description": "Open the quantum circuit playground",
        "url": "/#main-content",
        "icons": [{ "src": "/icons/playground-96.png", "sizes": "96x96" }]
      },
      {
        "name": "Documentation",
        "short_name": "Docs", 
        "description": "View documentation and tutorials",
        "url": "/#docs",
        "icons": [{ "src": "/icons/docs-96.png", "sizes": "96x96" }]
      }
    ]
  };

  React.useEffect(() => {
    // Dynamically inject manifest
    const manifestBlob = new Blob([JSON.stringify(manifest)], { 
      type: 'application/json' 
    });
    const manifestURL = URL.createObjectURL(manifestBlob);
    
    const link = document.createElement('link');
    link.rel = 'manifest';
    link.href = manifestURL;
    document.head.appendChild(link);

    // Add Apple touch icons
    const appleTouchIcon = document.createElement('link');
    appleTouchIcon.rel = 'apple-touch-icon';
    appleTouchIcon.href = '/icons/qmlab-180.png';
    appleTouchIcon.sizes = '180x180';
    document.head.appendChild(appleTouchIcon);

    return () => {
      document.head.removeChild(link);
      document.head.removeChild(appleTouchIcon);
      URL.revokeObjectURL(manifestURL);
    };
  }, []);

  return null;
};