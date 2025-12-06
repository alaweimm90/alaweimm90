import { FullConfig, chromium } from '@playwright/test'

export default async function globalSetup(config: FullConfig) {
  const browser = await chromium.launch()

  // Regular user storage state
  const userContext = await browser.newContext()
  await userContext.addCookies([
    {
      name: 'auth_token',
      value: 'user-token',
      domain: 'localhost',
      path: '/',
      httpOnly: true,
      secure: false,
      sameSite: 'Lax',
      expires: Math.floor(Date.now() / 1000) + 86400,
    },
  ])
  await userContext.storageState({ path: 'tests/e2e/storage/user.json' })
  await userContext.close()

  // Admin storage state
  const adminContext = await browser.newContext()
  await adminContext.addCookies([
    {
      name: 'auth_token',
      value: 'admin-token',
      domain: 'localhost',
      path: '/',
      httpOnly: true,
      secure: false,
      sameSite: 'Lax',
      expires: Math.floor(Date.now() / 1000) + 86400,
    },
  ])
  await adminContext.storageState({ path: 'tests/e2e/storage/admin.json' })
  await adminContext.close()

  await browser.close()
}