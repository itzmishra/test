import { motion } from 'framer-motion';
import ProjectCard from '../components/ProjectCard';
import SectionTitle from '../components/SectionTitle';
import projectsData from '../data/projects.json';

const Projects = () => {
  return (
    <div className="min-h-screen pt-20">
      <section className="section-padding bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
        <div className="container-custom">
          <SectionTitle
            title="My Projects"
            subtitle="Featured Work"
            center
          />
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center text-gray-600 dark:text-gray-400 max-w-2xl mx-auto mt-6"
          >
            A collection of my engineering projects spanning robotics, machine learning, and Formula Student competitions.
          </motion.p>
        </div>
      </section>

      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {projectsData.projects.map((project, index) => (
              <ProjectCard key={project.id} project={project} index={index} />
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Projects;
