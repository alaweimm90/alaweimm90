import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { ThemeProvider } from '@/components/providers/theme-provider'
import { QueryProvider } from '@/components/providers/query-provider'
import { Toaster } from '@/components/ui/toaster'
import { AuthProvider } from '@/components/providers/auth-provider'
import { SocketProvider } from '@/components/providers/socket-provider'
import { ErrorBoundary } from '@/components/error-boundary'
import { Analytics } from '@/components/analytics'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: 'Ghost Researcher - AI-Powered Research Assistant',
  description: 'Transform your research with AI-powered literature review, hypothesis generation, and paper writing assistance.',
  keywords: ['research', 'AI', 'literature review', 'academic', 'scientific research', 'paper writing'],
  authors: [{ name: 'Ghost Researcher Team' }],
  openGraph: {
    title: 'Ghost Researcher - AI-Powered Research Assistant',
    description: 'Transform your research with AI-powered literature review, hypothesis generation, and paper writing assistance.',
    url: 'https://ghostresearcher.com',
    siteName: 'Ghost Researcher',
    images: [
      {
        url: 'https://ghostresearcher.com/og-image.png',
        width: 1200,
        height: 630,
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Ghost Researcher - AI-Powered Research Assistant',
    description: 'Transform your research with AI-powered literature review, hypothesis generation, and paper writing assistance.',
    images: ['https://ghostresearcher.com/twitter-image.png'],
  },
  manifest: '/manifest.json',
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
  },
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: '#0F172A' },
  ],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
      </head>
      <body className={`${inter.variable} font-sans antialiased`}>
        <ErrorBoundary>
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
          >
            <AuthProvider>
              <QueryProvider>
                <SocketProvider>
                  {children}
                  <Toaster />
                  <Analytics />
                </SocketProvider>
              </QueryProvider>
            </AuthProvider>
          </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  )
}