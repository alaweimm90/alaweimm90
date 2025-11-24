import type { Request, Response, NextFunction } from 'express';

/**
 *
 * @param req
 * @param res
 * @param next
 */
export function authenticate(req: Request, res: Response, next: NextFunction): void {
  const requiredKey = process.env.COACHING_API_KEY;

  if (!requiredKey) {
    // No key configured: treat as development mode.
    next();
    return;
  }

  const apiKey = req.header('x-api-key');
  if (!apiKey || apiKey !== requiredKey) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  next();
}

/**
 *
 * @param allowedRoles
 */
export function authorizeRole(allowedRoles: string[]) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const role = (req.header('x-role') || '').toLowerCase();

    if (!role || !allowedRoles.includes(role)) {
      res.status(403).json({ error: 'Forbidden' });
      return;
    }

    next();
  };
}
