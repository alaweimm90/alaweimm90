import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main(): Promise<void> {
  console.log('Seeding database...');

  // Create sample users
  const user1 = await prisma.user.upsert({
    where: { email: 'admin@{{PROJECT_NAME}}.com' },
    update: {},
    create: {
      email: 'admin@{{PROJECT_NAME}}.com',
      name: 'Admin User',
      posts: {
        create: {
          title: 'Welcome to {{PROJECT_NAME}}',
          content: 'This is the first post.',
          published: true,
        },
      },
    },
  });

  console.log('Created user:', user1);
  console.log('Seeding complete.');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
