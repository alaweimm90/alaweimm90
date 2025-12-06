import swaggerJsdoc from 'swagger-jsdoc';

const options: swaggerJsdoc.Options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Live It Iconic API',
      version: '1.0.0',
      description:
        'E-commerce API for Live It Iconic - Premium automotive-inspired apparel and luxury merchandise',
      contact: {
        name: 'API Support',
        email: 'api@liveiconic.com',
        url: 'https://liveiconic.com/support',
      },
      license: {
        name: 'Proprietary',
      },
    },
    servers: [
      {
        url: 'http://localhost:5173',
        description: 'Development server',
      },
      {
        url: 'https://api.liveiconic.com',
        description: 'Production server',
      },
    ],
    components: {
      securitySchemes: {
        bearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT',
          description: 'JWT Bearer token authentication',
        },
      },
      schemas: {
        User: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Unique user identifier' },
            email: {
              type: 'string',
              format: 'email',
              description: 'User email address',
            },
            name: { type: 'string', description: 'User full name' },
            createdAt: {
              type: 'string',
              format: 'date-time',
              description: 'Account creation timestamp',
            },
          },
          required: ['id', 'email'],
        },
        Product: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Product identifier' },
            name: { type: 'string', description: 'Product name' },
            description: { type: 'string', description: 'Product description' },
            price: { type: 'number', description: 'Product price in USD' },
            category: {
              type: 'string',
              enum: ['tees', 'hoodies', 'caps', 'accessories'],
              description: 'Product category',
            },
            image: {
              type: 'string',
              format: 'uri',
              description: 'Product image URL',
            },
            inStock: {
              type: 'boolean',
              description: 'Stock availability',
            },
            stock: {
              type: 'integer',
              description: 'Number of items in stock',
            },
          },
          required: ['id', 'name', 'price', 'category'],
        },
        OrderItem: {
          type: 'object',
          properties: {
            productId: { type: 'string', description: 'Product identifier' },
            quantity: {
              type: 'integer',
              description: 'Item quantity',
            },
            price: { type: 'number', description: 'Price per unit' },
            subtotal: { type: 'number', description: 'Line item total' },
          },
          required: ['productId', 'quantity', 'price'],
        },
        Order: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Order identifier' },
            userId: { type: 'string', description: 'Customer user ID' },
            items: {
              type: 'array',
              items: { $ref: '#/components/schemas/OrderItem' },
              description: 'Order line items',
            },
            total: { type: 'number', description: 'Order total amount' },
            status: {
              type: 'string',
              enum: ['pending', 'processing', 'shipped', 'delivered', 'cancelled'],
              description: 'Order status',
            },
            createdAt: {
              type: 'string',
              format: 'date-time',
              description: 'Order creation timestamp',
            },
            shippingAddress: {
              type: 'object',
              description: 'Shipping address details',
            },
          },
          required: ['id', 'userId', 'items', 'total', 'status'],
        },
        Error: {
          type: 'object',
          properties: {
            error: { type: 'string', description: 'Error code' },
            message: { type: 'string', description: 'Error message' },
            details: {
              type: 'array',
              description: 'Additional error details',
            },
          },
          required: ['error', 'message'],
        },
        AuthResponse: {
          type: 'object',
          properties: {
            user: { $ref: '#/components/schemas/User' },
            accessToken: { type: 'string', description: 'JWT access token' },
            refreshToken: {
              type: 'string',
              description: 'JWT refresh token',
            },
          },
          required: ['user', 'accessToken'],
        },
        Inventory: {
          type: 'object',
          properties: {
            productId: { type: 'string' },
            available: { type: 'integer' },
            reserved: { type: 'integer' },
            total: { type: 'integer' },
          },
        },
        PaymentIntent: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            clientSecret: { type: 'string' },
            amount: { type: 'number' },
            currency: { type: 'string' },
            status: { type: 'string' },
          },
        },
      },
    },
    security: [{ bearerAuth: [] }],
  },
  apis: [
    './src/api/**/*.ts',
    './src/api/**/*.tsx',
  ],
};

export const swaggerSpec = swaggerJsdoc(options);
