import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import 'dotenv/config';
import OpenAI from 'openai';

const OUT = '.audit/ai';
fs.mkdirSync(OUT, { recursive: true });

const urls = [
  '/',
  '/docs',
  '/modules/physics'
];

const prompt = `
You are a senior product designer and accessibility expert.
Evaluate the screenshot for:
- Visual hierarchy, spacing, typography consistency, color/contrast, and responsiveness clues
- Accessibility issues (contrast, tap targets, focus order, landmarks)
- Brand cohesion and readability
- Concrete, prioritized fixes with rationale
Give a short score per area (0–100) and 5–10 prioritized actions.
`;

const run = async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  if (!process.env.OPENAI_API_KEY) {
    console.warn('OPENAI_API_KEY not set; skipping AI visual audit.');
    await browser.close();
    process.exit(0);
  }

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  for (const u of urls) {
    const url = `http://127.0.0.1:4173${u}`;
    await page.goto(url, { waitUntil: 'networkidle' });
    const buf = await page.screenshot({ fullPage: true });
    const png = await sharp(buf).png().toBuffer();
    const imgPath = path.join(OUT, `${u.replace(/[^a-z0-9]/gi,'_') || 'home'}.png`);
    fs.writeFileSync(imgPath, png);

    const b64 = png.toString('base64');
    const res = await client.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: 'Expert UI/UX designer and a11y auditor.' },
        { role: 'user', content: [
          { type: 'text', text: prompt },
          { type: 'image_url', image_url: { url: `data:image/png;base64,${b64}` } }
        ]}
      ],
      temperature: 0.2
    });

    const outPath = path.join(OUT, `${u.replace(/[^a-z0-9]/gi,'_') || 'home'}.md`);
    fs.writeFileSync(outPath, res.choices?.[0]?.message?.content || 'No response');
    console.log('AI review written to', outPath);
  }

  await browser.close();
};

run();

