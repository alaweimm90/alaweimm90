'use client'

import { createContext, useContext, useEffect, useState } from 'react'
import { io, Socket } from 'socket.io-client'
import { useAuth } from './auth-provider'

interface SocketContextType {
  socket: Socket | null
  connected: boolean
}

const SocketContext = createContext<SocketContextType | undefined>(undefined)

export function SocketProvider({ children }: { children: React.ReactNode }) {
  const [socket, setSocket] = useState<Socket | null>(null)
  const [connected, setConnected] = useState(false)
  const { user } = useAuth()

  useEffect(() => {
    if (user) {
      // In production, use actual WebSocket server URL
      const socketInstance = io(process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:3001', {
        auth: {
          token: localStorage.getItem('token'),
        },
      })

      socketInstance.on('connect', () => {
        console.log('Socket connected')
        setConnected(true)
      })

      socketInstance.on('disconnect', () => {
        console.log('Socket disconnected')
        setConnected(false)
      })

      setSocket(socketInstance)

      return () => {
        socketInstance.disconnect()
      }
    }
  }, [user])

  return (
    <SocketContext.Provider value={{ socket, connected }}>
      {children}
    </SocketContext.Provider>
  )
}

export function useSocket() {
  const context = useContext(SocketContext)
  if (context === undefined) {
    throw new Error('useSocket must be used within a SocketProvider')
  }
  return context
}