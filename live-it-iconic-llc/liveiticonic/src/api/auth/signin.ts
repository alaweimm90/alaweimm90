import bcrypt from 'bcrypt';
import { z } from 'zod';
import { getErrorMessage } from '@/types/api';
import { validate, signinSchema } from '@/middleware/validation';
import { generateAccessToken, generateRefreshToken } from '@/lib/jwt';
import { rateLimit } from '@/middleware/rateLimit';

/**
 * @swagger
 * /api/auth/signin:
 *   post:
 *     summary: Authenticate user
 *     description: Authenticates a user with email and password. Returns JWT access and refresh tokens.
 *     tags: [Authentication]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - email
 *               - password
 *             properties:
 *               email:
 *                 type: string
 *                 format: email
 *                 example: user@example.com
 *               password:
 *                 type: string
 *                 format: password
 *                 example: SecurePass123!
 *     responses:
 *       200:
 *         description: Authentication successful
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/AuthResponse'
 *       400:
 *         description: Validation error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *       401:
 *         description: Invalid credentials
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *       429:
 *         description: Too many requests - rate limit exceeded
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *       500:
 *         description: Server error during sign in
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // Apply rate limiting (stricter for signin - 5 requests per 15 minutes)
  const rateLimitResponse = rateLimit(req, {
    limit: 5,
    windowMs: 15 * 60 * 1000,
  });
  if (rateLimitResponse) {
    return rateLimitResponse;
  }

  try {
    const body = await req.json();

    // Validate input against schema
    const { email, password } = validate(signinSchema, body);

    // In production, this would:
    // 1. Query database for user by email
    // 2. Verify password against stored hashedPassword using bcrypt.compare()
    // For now, simulate successful signin

    // Simulate fetching user from database
    // const user = await db.users.findByEmail(email);
    // if (!user) {
    //   return new Response(JSON.stringify({ error: 'Invalid credentials' }), {
    //     status: 401,
    //     headers: { 'Content-Type': 'application/json' },
    //   });
    // }

    // Verify password (use bcrypt.compare in production)
    // const isValid = await bcrypt.compare(password, user.hashedPassword);
    // if (!isValid) {
    //   return new Response(JSON.stringify({ error: 'Invalid credentials' }), {
    //     status: 401,
    //     headers: { 'Content-Type': 'application/json' },
    //   });
    // }

    const userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const accessToken = generateAccessToken(userId);
    const refreshToken = generateRefreshToken(userId);

    const user = {
      id: userId,
      email,
      name: email.split('@')[0],
    };

    return new Response(
      JSON.stringify({
        user,
        accessToken,
        refreshToken,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    // Handle validation errors
    if (error instanceof z.ZodError) {
      return new Response(
        JSON.stringify({
          error: 'Validation failed',
          details: error.errors.map((e) => ({
            field: e.path.join('.'),
            message: e.message,
          })),
        }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Sign in failed',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
