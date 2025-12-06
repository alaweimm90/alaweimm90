'use client'

import { createContext, useContext, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'

interface User {
  id: string
  email: string
  name: string
  avatar?: string
  role: 'user' | 'admin'
}

interface AuthContextType {
  user: User | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  signup: (email: string, password: string, name: string) => Promise<void>
  updateUser: (user: Partial<User>) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    // Check for existing session
    const checkSession = async () => {
      try {
        const token = localStorage.getItem('token')
        if (token) {
          // In production, validate token with backend
          const userData = JSON.parse(localStorage.getItem('user') || '{}')
          setUser(userData)
        }
      } catch (error) {
        console.error('Session check failed:', error)
      } finally {
        setLoading(false)
      }
    }

    checkSession()
  }, [])

  const login = async (email: string, password: string) => {
    try {
      // In production, make API call to authenticate
      const mockUser: User = {
        id: '1',
        email,
        name: email.split('@')[0],
        role: 'user',
      }

      localStorage.setItem('token', 'mock-token')
      localStorage.setItem('user', JSON.stringify(mockUser))
      setUser(mockUser)
      router.push('/dashboard')
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  }

  const signup = async (email: string, password: string, name: string) => {
    try {
      // In production, make API call to create account
      const mockUser: User = {
        id: '1',
        email,
        name,
        role: 'user',
      }

      localStorage.setItem('token', 'mock-token')
      localStorage.setItem('user', JSON.stringify(mockUser))
      setUser(mockUser)
      router.push('/onboarding')
    } catch (error) {
      console.error('Signup failed:', error)
      throw error
    }
  }

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)
    router.push('/')
  }

  const updateUser = (updates: Partial<User>) => {
    if (user) {
      const updatedUser = { ...user, ...updates }
      setUser(updatedUser)
      localStorage.setItem('user', JSON.stringify(updatedUser))
    }
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        logout,
        signup,
        updateUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}