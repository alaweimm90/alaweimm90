import bcrypt from 'bcrypt';
import { z } from 'zod';
import { getErrorMessage } from '@/types/api';
import { validate, signupSchema } from '@/middleware/validation';
import { generateAccessToken } from '@/lib/jwt';
import { rateLimit } from '@/middleware/rateLimit';

/**
 * @swagger
 * /api/auth/signup:
 *   post:
 *     summary: Register a new user
 *     description: Creates a new user account with email and password. Password must be at least 8 characters with mixed case and numbers.
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
 *                 description: User email address
 *                 example: newuser@example.com
 *               password:
 *                 type: string
 *                 format: password
 *                 minLength: 8
 *                 description: Strong password (8+ chars, mixed case, numbers)
 *                 example: SecurePass123!
 *               name:
 *                 type: string
 *                 minLength: 2
 *                 description: Full name (optional)
 *                 example: John Doe
 *     responses:
 *       200:
 *         description: User created successfully
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
 *       429:
 *         description: Too many requests - rate limit exceeded
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 error:
 *                   type: string
 *                   example: Too many requests
 *       500:
 *         description: Server error during signup
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

  // Apply rate limiting
  const rateLimitResponse = rateLimit(req);
  if (rateLimitResponse) {
    return rateLimitResponse;
  }

  try {
    const body = await req.json();

    // Validate input against schema
    const { email, password, name } = validate(signupSchema, body);

    // Hash password with bcrypt (cost factor 12)
    const hashedPassword = await bcrypt.hash(password, 12);

    // In production, this would create user in database
    // Store both email and hashedPassword
    // For now, simulate successful signup
    const userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const accessToken = generateAccessToken(userId);

    const user = {
      id: userId,
      email,
      name: name || email.split('@')[0],
      createdAt: new Date().toISOString(),
    };

    return new Response(
      JSON.stringify({
        user,
        accessToken,
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
        error: getErrorMessage(error) || 'Sign up failed',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
